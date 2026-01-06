% Generates peridynamics operator learning datasets of type 1

% ===== Configuration =====

% Size of dataset
numSamples = 20;

% Spatial interval
x0 = 0;
xn = 1;
numCells = 128;

% Temporal interval
t0 = 0;
tn = 1;
dt = 0.01;

% Order of influence function
omegaOrder = 0;

% Peridynamic horizon
delta = 0.02;


% ===== Dataset generation =====

% Create PDProblem
problem = PDProblem();
problem.setUniformNodes(x0, xn, numCells);
problem.setUniformTimeSteps(t0, tn, dt);
problem.setInfluenceFunction(omegaOrder);
problem.delta = delta;

zero = zeros(size(problem.x));
problem.v0 = @(x) zero;
problem.bfunc = @(x, t) zero;

% Create solver
solver = PDSolver(problem);

% Create saver
saver = OLPDSaver1(problem, numSamples);

% Generate dataset
for i = 1:numSamples
    fprintf("Generating data sample %d\n", i);
    problem.u0 = @(x)
    saver.activeSample = i;
    solver.solve(@saver.stepCallback);
end