classdef PDFullSaver < handle
    % Saves the entire history of a peridynamics simulation
    %
    % Pass the stepCallback function to PDSolver.solve
    %

    properties

        % (num nodes, num times) vector of displacement field at all nodes
        % and times
        u
        
        % (num nodes, num times) vector of velocity field at all nodes and
        % times
        v

        % (num nodes, num times) vector of internal force density at all
        % nodes and all times
        f

        % (num nodes, num times) vector of body force density at all nodes
        % and all times
        b

        % (num nodes, num times) vector of macroelastic energy density at
        % all nodes and all times
        w

        % Whether to print progress updates on each step
        showProgress = false
    end

    methods
        function self = PDFullSaver(problem)
            % Create a PDFullSaver, optionally allocating space for known
            % PDProblem
            %
            % Parameters
            % ----------
            % problem: PDProblem, if provided output arrays will be
            %          pre-allocated with enough memory to store the entire
            %          history of the problem

            if nargin > 0
                self.allocate(problem)
            else
                warning("Creating PDFullSaver without problem information is slow. Call allocate(problem) to pre-allocate output variables.");
            end
        end

        function allocate(self, problem)
            % Allocates memory to store the entire history of the given
            % problem
            % 
            % This destroys any existing saved data.
            %
            % Parameters
            % ----------
            % problem: PDProblem for which to allocate memory

            self.u = zeros(problem.numNodes(), length(problem.t));
            self.v = zeros(size(self.u));
            self.f = zeros(size(self.u));
            self.b = zeros(size(self.u));
            self.w = zeros(size(self.u));
        end

        function stepCallback(self, k, u, v, f, b, w)
            % Callback to pass to PDSolver.solve
            % 
            % Parameters
            % ----------
            % k: Time step index, k = 1 means t0 to t1.
            % u: (num nodes, 1) column vector of current displacement field
            %    at each node
            % v: (num nodes, 1) column vector of current velocity field at
            %    each node
            % f: (num nodes, 1) column vector of current internal force 
            %    density at each node
            % b: (num nodes, 1) column vector of current body force density
            %    at each node
            
            self.u(:, k) = u;
            self.v(:, k) = v;
            self.f(:, k) = f;
            self.b(:, k) = b;
            self.w(:, k) = w;

            if self.showProgress
                fprintf("Simulation step %d\n", k);
            end
        end
    end
end