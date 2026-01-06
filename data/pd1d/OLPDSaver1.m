classdef OLPDSaver1 < handle
    % Saver for generating operator learning datasets of type 1
    % 
    % Saves only the initial and final displacement fields (initial
    % velocity is set to 0)
    %

    properties
        % This dataset uses the same PDProblem for all data points, so
        % store the problem here
        problem

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
            
            self.problem = problem;
            self.numTimeSteps = length(problem.t);
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

        function writeDataset(self, fileName)
            % Writes the saved data to an operator learning dataset file
            %
            % Parameters
            % ----------
            % fileName: Name of the output dataset file (.ol.h5)
            %
            % Overwrites fileName file with an operator learning dataset
            % containing the data saved on this object
            if isfile(fileName)
                delete(fileName);
            end
            
            % ***NOTE***: Make sure to write *all* arrays backwards because
            % MATLAB uses FORTRAN ordering, but NumPy uses C ordering

            % Create discretization information (x and y)
            numNodes = self.problem.numNodes();
            h5create(fileName, '/x/1', [1, numNodes], 'Datatype', 'single');
            h5write(fileName, '/x/1', single(self.problem.x'));
            h5writeatt(fileName, '/x/1', 'id', 1);
            
            h5create(fileName, '/y/1', [1, numNodes], 'Datatype', 'single');
            h5write(fileName, '/y/1', single(self.problem.x'));
            h5writeatt(fileName, '/y/1', 'id', 1);

            % Save u and v
            numSamples = size(self.u0, 1);
            indices = int32(0 : (numSamples-1));
            
            h5create(fileName, '/u/1/u', [1, numNodes, numSamples], 'Datatype', 'single');
            h5write(fileName, '/u/1/u', single(reshape(self.u0', 1, numNodes, numSamples)));
            h5create(fileName, '/u/1/indices', numSamples, 'Datatype', 'int32');
            h5write(fileName, '/u/1/indices', indices);
            h5writeatt(fileName, '/u/1', 'disc_id', 1);

            h5create(fileName, '/v/1/v', [1, numNodes, numSamples], 'Datatype', 'single');
            h5write(fileName, '/v/1/v', single(reshape(self.un', 1, numNodes, numSamples)));
            h5create(fileName, '/v/1/indices', numSamples, 'Datatype', 'int32');
            h5write(fileName, '/v/1/indices', indices);
            h5writeatt(fileName, '/v/1', 'disc_id', 1);
        end
    end
end