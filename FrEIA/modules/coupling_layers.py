from torch import Tensor

from . import InvertibleModule

from typing import Callable, Union


from math import exp

import torch
import torch.nn as nn

from FrEIA.modules.orthogonal import HouseholderPerm
from torch.nn import functional as F


class MCDropout(nn.Dropout):
    def __init__(self, p: float = 0.5, active: bool = True) -> None:
        super(MCDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.active = active

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.active)

    def set_active(self, active):
        self.active = active

class _BaseCouplingBlock(InvertibleModule):
    '''Base class to implement various coupling schemes.  It takes care of
    checking the dimensions, conditions, clamping mechanism, etc.
    Each child class only has to implement the _coupling1 and _coupling2 methods
    for the left and right coupling operations.
    (In some cases below, forward() is also overridden)
    '''

    def __init__(self, dims_in, dims_c=[],
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN"):
        '''
        Additional args in docstring of base class.

        Args:
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c)

        self.channels = dims_in[0][0]

        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])

        self.split_len1 = self.channels // 2
        self.split_len2 = self.channels - self.channels // 2

        self.clamp = clamp

        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        self.condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = lambda u: 0.636 * torch.atan(u)
            elif clamp_activation == "TANH":
                self.f_clamp = torch.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = lambda u: 2. * (torch.sigmoid(u) - 0.5)
            elif clamp_activation == "I":
                self.f_clamp = lambda u: u
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations
        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=-1)

        if not rev:
            x2_c = torch.cat([x2, *c], -1) if self.conditional else x2
            y1, j1 = self._coupling1(x1, x2_c)

            y1_c = torch.cat([y1, *c], -1) if self.conditional else y1
            y2, j2 = self._coupling2(x2, y1_c)

        else:
            # names of x and y are swapped for the reverse computation
            x1_c = torch.cat([x1, *c], -1) if self.conditional else x1
            y2, j2 = self._coupling2(x2, x1_c, rev=True)

            y2_c = torch.cat([y2, *c], -1) if self.conditional else y2
            y1, j1 = self._coupling1(x1, y2_c, rev=True)

        return (torch.cat((y1, y2), -1),), j1 + j2

    def _coupling1(self, x1, u2, rev=False):
        '''The first/left coupling operation in a two-sided coupling block.

        Args:
          x1 (Tensor): the 'active' half being transformed.
          u2 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y1 (Tensor): same shape as x1, the transformed 'active' half.
          j1 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        '''
        raise NotImplementedError()

    def _coupling2(self, x2, u1, rev=False):
        '''The second/right coupling operation in a two-sided coupling block.

        Args:
          x2 (Tensor): the 'active' half being transformed.
          u1 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y2 (Tensor): same shape as x1, the transformed 'active' half.
          j2 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        '''
        raise NotImplementedError()

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims


class NICECouplingBlock(_BaseCouplingBlock):
    '''Coupling Block following the NICE (Dinh et al, 2015) design.
    The inputs are split in two halves. For 2D, 3D, 4D inputs, the split is
    performed along the channel dimension. Then, residual coefficients are
    predicted by two subnetworks that are added to each half in turn.
    '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: callable = None):
        '''
        Additional args in docstring of base class.

        Args:
          subnet_constructor:
            Callable function, class, or factory object, with signature
            constructor(dims_in, dims_out). The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples.
            Two of these subnetworks will be initialized inside the block.
        '''
        super().__init__(dims_in, dims_c, clamp=0., clamp_activation=(lambda u: u))

        self.F = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1)
        self.G = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2)

    def _coupling1(self, x1, u2, rev=False):
        if rev:
            return x1 - self.F(u2), 0.
        return x1 + self.F(u2), 0.

    def _coupling2(self, x2, u1, rev=False):
        if rev:
            return x2 - self.G(u1), 0.
        return x2 + self.G(u1), 0.


class RNVPCouplingBlock(_BaseCouplingBlock):
    '''Coupling Block following the RealNVP design (Dinh et al, 2017) with some
    minor differences. The inputs are split in two halves. For 2D, 3D, 4D
    inputs, the split is performed along the channel dimension. For
    checkerboard-splitting, prepend an i_RevNet_downsampling module. Two affine
    coupling operations are performed in turn on both halves of the input.
    '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN"):
        '''
        Additional args in docstring of base class.

        Args:
          subnet_constructor: function or class, with signature
            constructor(dims_in, dims_out).  The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples. Four of these subnetworks will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        self.subnet_s1 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2)
        self.subnet_t1 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2)
        self.subnet_s2 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1)
        self.subnet_t2 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1)

    def _coupling1(self, x1, u2, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian
        s2 = self.subnet_s2(u2)
        t2 = self.subnet_t2(u2)
        s2 = self.clamp * self.f_clamp(s2)

        j1 = torch.sum(s2, dim=tuple(range(s2.ndim-1, self.ndims + 1)))

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            self.j1 = -j1
            return y1, -j1
        else:
            y1 = torch.exp(s2) * x1 + t2
            self.j1 = j1
            return y1, j1

    def _coupling2(self, x2, u1, rev=False):
        s1, t1 = self.subnet_s1(u1), self.subnet_t1(u1)
        s1 = self.clamp * self.f_clamp(s1)
        j2 = torch.sum(s1, dim=tuple(range(s1.ndim-1, self.ndims + 1)))

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            self.j2 = -j2
            return y2, -j2
        else:
            y2 = torch.exp(s1) * x2 + t1
            self.j2 = j2
            return y2, j2

    def jacobian(self):
        return self.j1 + self.j2


class GLOWCouplingBlock(_BaseCouplingBlock):
    '''Coupling Block following the GLOW design. Note, this is only the coupling
    part itself, and does not include ActNorm, invertible 1x1 convolutions, etc.
    See AllInOneBlock for a block combining these functions at once.
    The only difference to the RNVPCouplingBlock coupling blocks
    is that it uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate
    subnetworks. This reduces computational cost and speeds up learning.
    '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN"):
        '''
        Additional args in docstring of base class.

        Args:
          subnet_constructor: function or class, with signature
            constructor(dims_in, dims_out).  The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples. Two of these subnetworks will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        self.subnet1 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * 2)
        self.subnet2 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * 2)

    def _coupling1(self, x1, u2, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a2 = self.subnet2(u2)
        s2, t2 = torch.split(a2, [self.split_len1, self.split_len2], -1)
        s2 = self.clamp * self.f_clamp(s2)
        j1 = torch.sum(s2, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, -j1
        else:
            y1 = torch.exp(s2) * x1 + t2
            return y1, j1

    def _coupling2(self, x2, u1, rev=False):
        a1 = self.subnet1(u1)
        s1, t1 = torch.split(a1, [self.split_len2, self.split_len1], -1)
        s1 = self.clamp * self.f_clamp(s1)
        j2 = torch.sum(s1, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, -j2
        else:
            y2 = torch.exp(s1) * x2 + t1
            return y2, j2


class GINCouplingBlock(_BaseCouplingBlock):
    '''Coupling Block following the GIN design. The difference from
    GLOWCouplingBlock (and other affine coupling blocks) is that the Jacobian
    determinant is constrained to be 1.  This constrains the block to be
    volume-preserving. Volume preservation is achieved by subtracting the mean
    of the output of the s subnetwork from itself.  While volume preserving, GIN
    is still more powerful than NICE, as GIN is not volume preserving within
    each dimension.
    Note: this implementation differs slightly from the originally published
    implementation, which scales the final component of the s subnetwork so the
    sum of the outputs of s is zero. There was no difference found between the
    implementations in practice, but subtracting the mean guarantees that all
    outputs of s are at most ±exp(clamp), which might be more stable in certain
    cases.
    '''
    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN"):
        '''
        Additional args in docstring of base class.

        Args:
          subnet_constructor: function or class, with signature
            constructor(dims_in, dims_out).  The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples. Two of these subnetworks will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        self.subnet1 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * 2)
        self.subnet2 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * 2)

    def _coupling1(self, x1, u2, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a2 = self.subnet2(u2)
        s2, t2 = torch.split(a2, [self.split_len1, self.split_len2], -1)
        s2 = self.clamp * self.f_clamp(s2)
        s2 -= s2.mean(1, keepdim=True)

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, 0.
        else:
            y1 = torch.exp(s2) * x1 + t2
            return y1, 0.

    def _coupling2(self, x2, u1, rev=False):
        a1 = self.subnet1(u1)
        s1, t1 = torch.split(a1, [self.split_len2, self.split_len1], -1)
        s1 = self.clamp * self.f_clamp(s1)
        s1 -= s1.mean(1, keepdim=True)

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, 0.
        else:
            y2 = torch.exp(s1) * x2 + t1
            return y2, 0.


class AffineCouplingOneSided(_BaseCouplingBlock):
    '''Half of a coupling block following the GLOWCouplingBlock design.  This
    means only one affine transformation on half the inputs.  In the case where
    random permutations or orthogonal transforms are used after every block,
    this is not a restriction and simplifies the design.  '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN"):
        '''
        Additional args in docstring of base class.

        Args:
          subnet_constructor: function or class, with signature
            constructor(dims_in, dims_out).  The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples. One subnetwork will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation)
        self.subnet = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2)

    def forward(self, x, c=[], rev=False, jac=True):
        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=-1)
        x1_c = torch.cat([x1, *c], -1) if self.conditional else x1

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        s, t = self.subnet(x1_c), self.subnet(x1_c)
        s = self.clamp * self.f_clamp(s)
        j = torch.sum(s, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y2 = (x2 - t) * torch.exp(-s)
            j *= -1
        else:
            y2 = x2 * torch.exp(s) + t

        return (torch.cat((x1, y2), -1),), j


class ConditionalAffineTransform(_BaseCouplingBlock):
    '''Similar to the conditioning layers from SPADE (Park et al, 2019): Perform
    an affine transformation on the whole input, where the affine coefficients
    are predicted from only the condition.
    '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN"):
        '''
        Additional args in docstring of base class.

        Args:
          subnet_constructor: function or class, with signature
            constructor(dims_in, dims_out).  The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples. One subnetwork will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        if not self.conditional:
            raise ValueError("ConditionalAffineTransform must have a condition")

        self.subnet = subnet_constructor(self.condition_length, self.channels)

    def forward(self, x, c=[], rev=False, jac=True):
        if len(c) > 1:
            cond = torch.cat(c, -1)
        else:
            cond = c[0]

        # notation:
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        s = self.subnet(cond)
        t = self.subnet(cond)
        s = self.clamp * self.f_clamp(s)
        j = torch.sum(s, dim=tuple(range(s.ndim-1, self.ndims + 1)))

        if rev:
            y = (x[0] - t) * torch.exp(-s)
            self.jac = -j
            return (y,), -j
        else:
            y = torch.exp(s) * x[0] + t
            self.jac = j
            return (y,), j

    def jacobian(self):
        return self.jac


def linear_subnet_constructor(c_in, c_out, c_internal):
    return nn.Sequential(nn.Linear(c_in,       c_internal), nn.ReLU(),
                         nn.Linear(c_internal, c_internal), nn.ReLU(),
                         nn.Linear(c_internal, c_out))


def conv_subnet_constructor(c_in, c_out, c_internal):
    return nn.Sequential(nn.Conv2d(c_in,       c_internal, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(c_internal, c_internal, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(c_internal, c_out,      3, padding=1))


class HierarchicalAffineCouplingTree(_BaseCouplingBlock):

    def __init__(self, dims_in, dims_c, conv=False, subnet_constructor=None, c_internal=[], clamp=2.0,
                 clamp_activation: Union[str, Callable] = "ATAN", max_splits=-1, min_split_size=2, reshuffle=False):
        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        if subnet_constructor is None:
            subnet_constructor = conv_subnet_constructor if conv else linear_subnet_constructor
        if len(c_internal) == 0:
            c_internal = [self.channels,]
        if len(c_internal) == 1:
            c_internal += c_internal

        if reshuffle:
            self.perm = HouseholderPerm(dims_in, dims_c=dims_c, n_reflections=self.channels, fixed=True)
        else:
            self.perm = None

        self.s = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2, c_internal[0])
        self.t = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2, c_internal[0])

        if dims_in[0][0] >= 2 * min_split_size and max_splits != 0:
            self.leaf = False
            self.upper = HierarchicalAffineCouplingTree([(self.split_len1,)] + dims_in[1:], dims_c,
                                                        conv, subnet_constructor, c_internal[1:], clamp, max_splits-1, min_split_size, reshuffle)
            self.lower = HierarchicalAffineCouplingTree([(self.split_len2,)] + dims_in[1:], dims_c,
                                                        conv, subnet_constructor, c_internal[1:], clamp, max_splits-1, min_split_size, reshuffle)
        else:
            self.leaf = True

    def forward(self, x, c=[], rev=False, jac=True):
        # Potentially reshuffle
        if not rev and self.perm is not None:
            x = self.perm(x)[0]

        # Split data lanes
        x_upper, x_lower = torch.split(x[0], [self.split_len1, self.split_len2], dim=-1)

        if (not self.leaf) and (not rev):
            # Recursively run subtree transformations
            x_upper, J_upper = self.upper.forward([x_upper], c=c, rev=rev)
            x_lower, J_lower = self.lower.forward([x_lower], c=c, rev=rev)
            x_upper = x_upper[0]
            x_lower = x_lower[0]

        # Compute own coupling transform and Jacobian
        x_upper_c = torch.cat([x_upper, *c], dim=-1) if self.conditional else x_upper
        s, t = self.s(x_upper_c), self.t(x_upper_c)

        def clamping(s, factor, func="I"):
            if isinstance(func, str):
                if func == "ATAN":
                    f_clamp = lambda u: 0.636 * torch.atan(u)
                elif func == "TANH":
                    f_clamp = torch.tanh
                elif func == "SIGMOID":
                    f_clamp = lambda u: 2. * (torch.sigmoid(u) - 0.5)
                elif func == "I":
                    f_clamp = lambda u: u
                else:
                    raise ValueError(f'Unknown clamp activation "{func}"')
            else:
                f_clamp = func
            return factor * f_clamp(s)
        s = clamping(s, self.clamp)
        J = torch.sum(s, dim=tuple(range(s.ndim-1, self.ndims + 1)))
        if not rev:
            x_lower = torch.exp(s) * x_lower + t
        else:
            x_lower = (x_lower - t) * torch.exp(-s)
            J = J * (-1)

        if (not self.leaf) and rev:
            # Reverse order of hierarchy during inverse pass
            x_upper, J_upper = self.upper.forward([x_upper], c=c, rev=rev)
            x_lower, J_lower = self.lower.forward([x_lower], c=c, rev=rev)
            x_upper = x_upper[0]
            x_lower = x_lower[0]

        x = (torch.cat([x_upper, x_lower], dim=-1), )

        # Potentially reverse reshuffling
        if rev and self.perm is not None:
            x = self.perm(x, rev=True)[0]

        if not self.leaf:
            J = J_upper + J + J_lower

        return x, J


class HierarchicalAffineCouplingBlock(_BaseCouplingBlock):

    def __init__(self, dims_in, dims_c=[], conv=False, subnet_constructor=None, c_internal=[], clamp=4.,
                 clamp_activation: Union[str, Callable] = "ATAN", max_splits=-1, min_split_size=2, reshuffle=False):
        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        self.tree = HierarchicalAffineCouplingTree(dims_in,
                                                   dims_c=dims_c,
                                                   conv=conv,
                                                   subnet_constructor=subnet_constructor,
                                                   c_internal=c_internal,
                                                   clamp=clamp,
                                                   clamp_activation=clamp_activation,
                                                   max_splits=max_splits,
                                                   min_split_size=min_split_size,
                                                   reshuffle=reshuffle)

    def forward(self, x, c=[], rev=False, jac=True):
        x, self.jac = self.tree.forward(x, c, rev=rev, jac=jac)
        if jac:
            return x, self.jac
        else:
            return x

    def jacobian(self):
        return self.jac