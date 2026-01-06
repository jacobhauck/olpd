% Simple test and demonstration of how to use the solver

% Set up problem with Gaussian initial displacement and zero velocity and
% body force
problem = PDProblem();
zero = zeros(size(problem.x));
problem.u0 = @(x) 0.02 * exp(-(x - 0.5).^2/(2*0.05^2));
problem.v0 = @(x) zero;
problem.bfunc = @(x, t) zero;

% Define saving protocol with PDFullSaver to save all fields at all time
% steps in memory
saver = PDFullSaver(problem);
saver.showProgress = true;

% Create and run solver
solver = PDSolver(problem);
solver.solve(@saver.stepCallback);
