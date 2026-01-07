% Generates peridynamics operator learning datasets of type 1

% ===== Configuration =====

% Output file
fileName = "test3.ol.h5";

% Whether to show an example before generating the dataset
showExample = false;

% Size of dataset
numSamples = 10;

% Spatial interval
x0 = 0;
xn = 1;
numCells = 256;

% Temporal interval
t0 = 0;
tn = 1;
dt = 0.001;

% Order of influence function
omegaOrder = 0;

% Peridynamic horizon
delta = 0.02;

% Input GRF parameters
grfNumModes = 32;
grfMean = 0;
grfGamma = 1.4;
grfTau = 1;
grfSigma= 0.2;
grfType = "neumann";


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

% Show an example, if requested
if showExample
    u0ref = GRF(grfNumModes, grfMean, grfGamma, grfTau, grfSigma, grfType);
    problem.u0 = @(x) u0ref((x - x0) / (xn - x0));
    exampleSaver = PDFullSaver(problem);
    solver.solve(@exampleSaver.stepCallback);
    
    h = animatedline;
    xlabel("x");
    ylabel("u(x, t)");
    ylim([min(min(exampleSaver.u)), max(max(exampleSaver.u))]);
    for k = 1:length(problem.t)
        clearpoints(h);
        addpoints(h, problem.x, exampleSaver.u(:, k));
        drawnow;
    end
end

% Generate dataset
prog = waitbar(0, "Starting dataset generation");
for i = 1:numSamples
    u0ref = GRF(grfNumModes, grfMean, grfGamma, grfTau, grfSigma, grfType);
    problem.u0 = @(x) u0ref((x - x0) / (xn - x0));
    saver.activeSample = i;
    solver.solve(@saver.stepCallback);
    waitbar(i / numSamples, prog, sprintf("Generated %d / %d", i, numSamples));
end

waitbar(1.0, prog, "Saving...");
saver.writeDataset(fileName);
close(prog);
