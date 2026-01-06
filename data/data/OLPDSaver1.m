classdef OLPDSaver1 < handle
    % Saver for generating operator learning datasets of type 1
    % 
    % Saves only the initial and final displacement fields (initial
    % velocity is set to 0)
    %

    properties
        % int, which sample in the dataset is being written
        activeSample = 1

        % How many time steps are in each simulation
        numTimeSteps

        % (num samples, num nodes) vector of initial displacement field at 
        % all nodes and for all samples 
        u0
        
        % (num samples, num nodes) vector of final displacement field at 
        % all nodes and for all samples 
        un
        
        % Whether to print progress updates on each step
        showProgress = false
    end

    methods
        function self = OLPDSaver1(problem, numSamples)
            % Create a PDFullSaver, optionally allocating space for known
            % PDProblem
            %
            % Parameters
            % ----------
            % problem: PDProblem, if provided output arrays will be
            %          pre-allocated with enough memory to store the entire
            %          history of the problem
            % numSamples: Size of the operator learning dataset

            if nargin > 0
                self.numTimeSteps = length(problem.t);
                self.allocate(problem, numSamples);
            else
                warning("Creating PDFullSaver without problem information is slow. Call allocate(problem) to pre-allocate output variables.");
            end
        end

        function allocate(self, problem, numSamples)
            % Allocates memory to store the initial and final displacement
            % fields
            % 
            % This destroys any existing saved data.
            %
            % Parameters
            % ----------
            % problem: PDProblem for which to allocate memory
            
            self.u0 = zeros(numSamples, problem.numNodes());
            self.un = zeros(numSamples, problem.numNodes());
        end

        function stepCallback(self, k, u, ~, ~, ~, ~)
            % Callback to pass to PDSolver.solve
            % 
            % Parameters
            % ----------
            % k: Time step index, k = 1 means t0 to t1.
            % u: (num nodes, 1) column vector of current displacement field
            %    at each node
            
            if k == 1
                self.u0(self.activeSample, :) = u;
            elseif k == self.numTimeSteps
                self.un(self.activeSample, :) = u;
            end

            if self.showProgress
                fprintf("Simulation step %d\n", k);
            end
        end
    end
end