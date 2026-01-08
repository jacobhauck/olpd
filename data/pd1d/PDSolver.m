classdef PDSolver < handle
    % Solves a 1D peridynamics problem
    
    properties
        % PDProblem problem specification
        problem
        
        % Matrix of neighbor indices (num nodes, max neighbors)
        neighborIndices = [];

        % Matrix of neighbor signed distances (num nodes, max neighbors)
        neighborDistances

        % Matrix of neighbor influences (num nodes, max neighbors)
        neighborInfluences

        % Matrix of neighbor quadrature weights (num nodes, max neighbors)
        neighborWeights

        % Matrix indicating which neighbors are valid (num nodes, max
        % neighbors)
        neighborMask
    end

    methods
        function self = PDSolver(problem)
            if nargin > 0
                self.problem = problem;
                self.calculateNeighborInformation();
            end
        end

        function self = calculateNeighborInformation(self)
            % Computes neighbor information and caches it on this object.
            % 
            % Let num nodes = number of nodes = self.problem.numNodes()
            % and let max neighbors be the maximum number of neighbors
            % for any node.
            %
            % Outputs
            % -------
            % self.neighborIndices: (num nodes, max neighbors) matrix,
            %                       with self.neighborIndices(i, :) being
            %                       a row vector containing the indices
            %                       of the neighbors of node i
            % self.neighborDistances: (num nodes, max neighbors) matrix,
            %                         with self.neighborDistances(i, j)
            %                         being the signed distance between
            %                         nodes i and j, so x(j) - x(i)
            % self.neighborInfluences: (num nodes, max neighbors) matrix,
            %                          with self.neighborInfluences(i, j)
            %                          being the influence between nodes
            %                          i and j, so omega(|x(j) - x(i)|)
            % self.neighborWeights: (num nodes, max neighbors) matrix,
            %                       with self.neighborWeights(i, j) being
            %                       the quadrature weight assigned to
            %                       node j in the numerical quadrature to
            %                       approximate the total internal force 
            %                       on node i
            % self.neighborMask: (num nodes, max neighbors) matrix, since
            %                    each node may have a different number of
            %                    neighbors, the above four matrices have
            %                    enough columns for the maximum number
            %                    of neighbors, but in the case node i has
            %                    fewer than the maximum, say n < max 
            %                    neighbors, only the first n columns of
            %                    row i in the above four matrices will be
            %                    used. This matrix is a mask indicating
            %                    which elements of the above matrices are
            %                    actually in use, so 
            %                    self.neighborMask(i, j) = true iff the
            %                    elements in row i, column j of the above
            %                    four matrices are being used.  

            % Finds indices of neighboring nodes
            numNodes = self.problem.numNodes();
            x = self.problem.x;
            neighborIndicesTemp = cell(numNodes);
            maxNeighbors = 0;

            % Find indices
            for i = 1 : numNodes
                x_i = x(i);
                iLeft = i;
                iRight = i;
                
                % Look left
                for j = (i - 1) : (-1) : 1
                    if x_i - x(j) <= self.problem.delta
                        iLeft = j;
                    end
                end
                
                % Look right
                for j = (i + 1) : numNodes
                    if x(j) - x_i <= self.problem.delta
                        iRight = j;
                    end
                end
                
                % Collect indices
                leftInds = (iLeft : i - 1)';
                rightInds = (i + 1 : iRight)';
                neighborIndicesTemp{i} = [leftInds; rightInds];
                maxNeighbors = max(maxNeighbors, length(neighborIndicesTemp{i}));
            end
            
            % Initialize neighbor data
            self.neighborIndices = ones(numNodes, maxNeighbors);
            self.neighborDistances = ones(numNodes, maxNeighbors);
            self.neighborWeights = zeros(numNodes, maxNeighbors);
            self.neighborMask = false(numNodes, maxNeighbors);
            
            boundaries = self.problem.calcCellBoundaries();

            % Write neighbor data
            for i = 1 : numNodes
                numNeighbors = length(neighborIndicesTemp{i});
                indices = neighborIndicesTemp{i};
                self.neighborIndices(i, 1:numNeighbors) = indices;
                self.neighborMask(i, 1:numNeighbors) = true;
                
                % Calculate intersection of node cells with peridynamic
                % neighborhood for IPA-AC
                a = max(x(i) - self.problem.delta, boundaries(indices));
                b = min(x(i) + self.problem.delta, boundaries(indices + 1));

                % Use distance from centroid of intersection
                centroid = (a + b) / 2;
                self.neighborDistances(i, 1:numNeighbors) = centroid - x(i);

                % and area of intersection
                self.neighborWeights(i, 1:numNeighbors) = b - a;
            end

            % Precompute influence function for each bond
            r = abs(self.neighborDistances);
            self.neighborInfluences = self.problem.omega(r);
        end

        function [forceDensity, energyDensity] = calculateInternalForceDensity(self, u)
            % Calculates the internal force density and macroelastic energy 
            % density for a given displacement field
            %
            % Parameters
            % ----------
            % u: (num nodes, 1) column vector giving the displacement for
            %    each node
            %
            % Outputs
            % -------
            % forceDensity: (num nodes, 1) force density at each node for 
            %               the given displacement
            % energyDensity: (num nodes, 1) energy density at each node for
            %                the given displacement

            % Get number of neighbors and cross-sectional area
            maxNeighbors = size(self.neighborWeights, 2);
            a = self.problem.area;

            % Calculate force density f
            uDist = u(self.neighborIndices) - repmat(u, 1, maxNeighbors);
            curDist = uDist + self.neighborDistances;
            absDist = abs(self.neighborDistances);

            stretch = (abs(curDist) - absDist) ./ absDist;
            
            c = self.problem.getMicromodulusConstant();
            bondForces = c * (sign(curDist) .* self.neighborInfluences .* stretch);
            forceDensity = a * sum(bondForces .* self.neighborWeights, 2);
            
            % Calculate macroelastic energy density w
            bondEnergies = (c/2) * (self.neighborInfluences .* stretch.^2 .* absDist);
            energyDensity = (a/2) * sum(bondEnergies .* self.neighborWeights, 2);
        end

        function [uNext, vNext, fNext, bNext, wNext] = timeStep(self, k, u, v, f, b)
            % Perform one time step of the simulation
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
            %
            % Outputs
            % -------
            % uNext: (num nodes, 1) column vector of displacement field at
            %        the next time step (k+1) at each node
            % vNext: (num nodes, 1) column vector of velocity field at next
            %        time step (k+1) at each node
            % fNext: (num nodes, 1) column vector of internal force density
            %        at next time step (k+1) at each node
            % bNext: (num nodes, 1) column vector of body force density at
            %        next time step (k+1) at each node
            % wNext: (num nodes, 1) column vector of macroelastic energy
            %        density at next time step (k+1) at each node 

            dt = self.problem.getDeltaT(k);

            % Velocity Verlet method
            vHalf = v + (dt/2 / self.problem.rho) * (f + b);
            uNext = u + dt * vHalf;
            
            [fNext, wNext] = self.calculateInternalForceDensity(uNext);
            bNext = self.problem.bfunc(self.problem.x, self.problem.t(k+1));
            vNext = v + (dt/2 / self.problem.rho) * (fNext + bNext);
        end

        function solve(self, stepCallback)
            % Solves the peridynamics problem
            %
            % This uses cached neighbor information, so if you modified the
            % problem in such a way as to modify the neighbor information,
            % then please call calculateNeighborInformation() explicitly
            % to avoid using stale cached data (it would be too complicated
            % for me to keep track of this in the code, so be careful :D)
            %
            % Parameters
            % ----------
            % stepCallback: callback function called at each time sampling
            %               point in self.problem.t (so, including t0 and tn)
            %               with signature (k, u, v, f, b, w) -> void,
            %               where k, u, v, f, b, w are the same as the
            %               parameters described in timeStep
            
            if isempty(self.neighborIndices)
                self.calculateNeighborInformation();
            end

            % Initialize state
            u = self.problem.u0(self.problem.x);
            v = self.problem.v0(self.problem.x);
            b = self.problem.bfunc(self.problem.x, 0);
            [f, w] = self.calculateInternalForceDensity(u);
            stepCallback(1, u, v, f, b, w);
            
            % Perform time steps
            numSimSteps = length(self.problem.t) - 1;
            for k = 1 : numSimSteps
                [u, v, f, b, w] = self.timeStep(k, u, v, f, b);
                stepCallback(k + 1, u, v, f, b, w);
            end
        end
    end
end