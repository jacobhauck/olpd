classdef PDProblem < handle
    % Defines a 1D peridynamics problem
    % 
    % Used to configure the problem settings for PDSolver
    %

    properties
        % Lower boundary of domain
        x0 = 0.0

        % Upper boundary of domain
        xn = 1.0

        % Discrete spatial node locations (cell centers in increasing 
        % order--x must define valid cells in the domain [x0, xn])
        x = (1/256 : 1/128 : 255/256)'
        
        % Sampling points in time
        t = (0 : 1/500 : 1)'
        
        % Peridnynamic horizon
        delta = 0.02

        % Influence function (as callable)
        omega = @(r) ones(size(r))
        
        % Initial displacement field; callable with signature (x) -> (u),
        % where x and u are column vectors of the same dimension
        u0
        
        % Initial velocity (\dot{u}) field; callable with signature (x) ->
        % (v), where x and v are column vectors of the same dimension
        v0

        % Body force density function; callable with signature (x, t) ->
        % (b), where x and b are column vectors of the same dimension, and
        % t is a scalar
        bfunc

        % Density
        rho = 1.0

        % Cross-sectional area
        area = 0.05

        % Micromodulus constant
        c = 1.25e5;
    end

    methods
        function numNodes = numNodes(self)
            % Gets the number of nodes in the problem
            % 
            % Outputs
            % -------
            % numNodes: The number of nodes

            numNodes = length(self.x);
        end

        function self = setUniformNodes(self, x0, xn, n)
            % Sets node positions to centers of n uniform cells between
            % x0 and x1
            %
            % Parameters
            % ----------
            % x0: Leftmost node position
            % xn: Rightmost node position
            % n: Number of nodes
            % 
            % Outputs
            % -------
            % self.x0: is set to x0
            % self.xn: is set to xn
            % self.x: (n, 1) matrix, is set to the centers of the uniformly
            %         spaced node cells
            
            dx = (xn - x0) / n;
            self.x0 = x0;
            self.xn = xn;
            self.x = linspace(x0 + dx/2, xn - dx/2, n)';
        end

        function self = setUniformTimeSteps(self, t0, tn, dt)
            % Sets time sampling points to be uniform between t0 and tn
            % with a time step of dt
            % 
            % If (tn - t0) % dt ~= 0, then tn is replaced by tn' = t0 + the
            % smallest integer multiple of dt such that tn' < tn.
            % 
            % Parameters
            % ----------
            % t0: Start time
            % tn: Desired end time
            % dt: Time step

            self.t = (t0 : dt : tn)';
        end

        function self = setInfluenceFunction(self, order)
            % Sets the influence function to the default influence function
            % of the given order
            %
            % Parameters
            % ----------
            % order: Order of default influence function to set. Must be
            %        one of 0, 0.5, 1, 3, 5, or 7, otherwise an error is
            %        raised
            %
            % Outputs
            % -------
            % self.omega: is set to the default influence function of the
            %             given order

            d = self.delta;
            
            if order == 0
                self.omega = @(r) ones(size(r));
            elseif order == 0.5
                self.omega = @(r) 1.0 * (r <= d);
            elseif order == 1
                self.omega = @(r) (1.0 - r/d) * (r <= d);
            elseif order == 3
                self.omega = @(r) (1.0 - 3 * (r/d).^2 + 2 * (r/d).^3);
            elseif order == 5
                self.omega = @(r) (1.0 - 10 * (r/d).^3 + 15 * (r/d).^4 - 6 * (r/d).^5);
            elseif order == 7
                self.omega = @(r) (1.0 - 35 * (r/d).^4 + 84 * (r/d).^5 - 70 * (r/d).^6 + 20 * (r/d) .^ 7);
            else
                error("Invalid influence function order. Choose from 0, 0.5, 1, 3, 5 or 7.");
            end
        end

        function boundaries = calcCellBoundaries(self)
            % Calculates the boundaries of the node cells
            %
            % Error is raised if the cell centers and domain boundary
            % cannot define a valid set of cells.
            %
            % Outputs
            % -------
            % boundaries: (numNodes + 1, 1) column vector, which contains
            %             the boundaries of the cells in increasing order,
            %             so boundaries(i : i+1) gives the left and right
            %             cell boundary coordinates of node i.

            boundaries = zeros(self.numNodes() + 1, 1);
            boundaries(1) = self.x0;
            for i = 1 : self.numNodes()
                boundaries(i + 1) = 2 * self.x(i) - boundaries(i);

                if boundaries(i + 1) <= boundaries(i)
                    error("Invalid cell centers provided");
                end
            end

            if abs(boundaries(end) - self.xn) > 1e-8
                error("Invalid cell centers provided");
            end
        end

        function dt = getDeltaT(self, k)
            % Calculates time step size for the given simulation step
            %
            % Outputs
            % -------
            % dt: Time step size for step k (first step, from t0 to t1, is
            %     indicated by k = 1)

            dt = self.t(k + 1) - self.t(k);
        end
    end
end