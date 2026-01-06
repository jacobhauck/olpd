function u = GRF(num_modes, mean, gamma, tau, sigma, type)
% Generate a random function from the distribution N(mean, C) on [0 1] where
% C = sigma^2(-Delta + tau^2 I)^(-gamma), with a given type of boundary
% conditions (periodic, homogeneous Dirichlet, or homogeneous Neumann)
% 
% Parameters
% ----------
%   num_modes: number of Fourier modes to use. Generally grid size / 2
%   mean: mean of the GRF. If using Dirichlet boundary conditions, this parameter is
%         ignored and a mean of 0 is used.
%   gamma, tau, sigma: parameters of the covariance; see above
%   type: Type of boundary conditions. Must be on of "periodic", "dirichlet", or
%   "neumann".

if type == "dirichlet"
    mean = 0;
end

if type == "periodic"
    my_const = 2*pi;
else
    my_const = pi;
end

my_eigs = sqrt(2)*(abs(sigma).*((my_const.*(1:num_modes)').^2 + tau^2).^(-gamma/2));

if type == "dirichlet"
    alpha = zeros(num_modes,1);
else
    xi_alpha = randn(num_modes,1);
    alpha = my_eigs.*xi_alpha;
end

if type == "neumann"
    beta = zeros(num_modes,1);
else
    xi_beta = randn(num_modes,1);
    beta = my_eigs.*xi_beta;
end

a = alpha/2;
b = -beta/2;

c = [flipud(a) - flipud(b).*1i;mean + 0*1i;a + b.*1i];

if type == "periodic"
    uu = chebfun(c, [0 1], 'trig', 'coeffs');
    u = chebfun(@(t) uu(t - 0.5), [0 1], 'trig');
else
    uu = chebfun(c, [-pi pi], 'trig', 'coeffs');
    u = chebfun(@(t) uu(pi*t), [0 1]);
end