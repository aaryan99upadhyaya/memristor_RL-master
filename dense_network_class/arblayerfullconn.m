classdef arblayerfullconn < handle
    % The class of a simple multilayer perceptron (with backpropagation)
    % Optimizer : SGD, RMSprop
    % Backend: software, array obj
    % Activation (fixed): ReLU and linear (last layer)
    % Loss (fixed): MSE

    properties

        % Dimension of the network
        net_size

        % Optimizer
        optimizer

        % Weights and biases
        weights
        biases
        z
        a
        nabla_w
        nabla_b
        last_nabla_w
        last_nabla_b
        grad_mean_sqr_w
        grad_mean_sqr_b

        % Software or hardware
        software

        % Learning rate
        learningrate=0.0005;

        % Physical/simu array
        array_interface

        % Momentum
        momentum = 0.0;

        % RMSprop
        decay = 0.9;
        eps = 1e-8

        % Conductance model (all conductances in microsiemens)
        conductance_mode = false;
        G_min_uS = 0.1;
        G_max_uS = 150.0;
        G_noise_uS = 0.1;
        W_max = 2.5;

        % Inf = no quantization, 8 = 8-bit
        conductance_bits = Inf;

        % 'static' = fixed device variation
        % 'read' = new noise on every forward pass
        noise_mode = 'static';

        G_pos_uS
        G_neg_uS
        noise_pos_uS
        noise_neg_uS

        % Actual weights used during forward propagation
        forward_weights
    end

    methods
        %%

        function obj = arblayerfullconn(net_size, varargin)
            % Create the multilayer perceptron obj

            % Public MATLAB parser (no Statistics Toolbox dependency).
            p = inputParser;
            addParameter(p, 'optimizer', 'SGD');
            addParameter(p, 'software', false);
            addParameter(p, 'conductance_mode', false);
            addParameter(p, 'conductance_noise_uS', 0.1);
            addParameter(p, 'conductance_bits', Inf);
            addParameter(p, 'noise_mode', 'static');
            parse(p, varargin{:});

            obj.optimizer = p.Results.optimizer;
            obj.software = logical(p.Results.software);
            obj.conductance_mode = logical(p.Results.conductance_mode);
            obj.G_noise_uS = p.Results.conductance_noise_uS;
            obj.conductance_bits = p.Results.conductance_bits;
            obj.noise_mode = p.Results.noise_mode;

            validateattributes(obj.G_noise_uS, {'numeric'}, ...
                {'scalar','real','finite','nonnegative'});
            if ~(isnumeric(obj.conductance_bits) && ...
                    isscalar(obj.conductance_bits) && ...
                    isreal(obj.conductance_bits) && ...
                    (isinf(obj.conductance_bits) || ...
                    (isfinite(obj.conductance_bits) && ...
                     obj.conductance_bits >= 1 && ...
                     obj.conductance_bits == floor(obj.conductance_bits))))
                error('conductance_bits must be Inf or a positive integer.');
            end
            if ~any(strcmp(obj.noise_mode, {'static','read'}))
                error('noise_mode must be ''static'' or ''read''.');
            end
            if obj.conductance_mode && ~obj.software
                error(['conductance_mode is a software simulation; ' ...
                    'the hardware backend is not implemented here.']);
            end
            if obj.G_min_uS >= obj.G_max_uS || obj.W_max <= 0
                error('Invalid conductance range or weight scale.');
            end

            % Dimension of the network
            obj.net_size = net_size; % net_size: column vector, e.g. [4;64;2]

            % Link with array object
            %obj.array_interface = array_interface;

            % Assign dimensions to properties of the class
            obj.weights = cell(length(obj.net_size)-1,1);
            obj.biases = obj.weights;
            obj.last_nabla_w = obj.weights;
            obj.last_nabla_b = obj.weights;
            obj.z = obj.weights;
            obj.a = obj.weights;
            obj.nabla_w = obj.weights;
            obj.nabla_b = obj.weights;
            obj.grad_mean_sqr_w = obj.weights;
            obj.grad_mean_sqr_b = obj.weights;

            % Conductance and device-variation matrices, one per layer.
            obj.G_pos_uS = cell(length(net_size)-1, 1);
            obj.G_neg_uS = cell(length(net_size)-1, 1);
            obj.noise_pos_uS = cell(length(net_size)-1, 1);
            obj.noise_neg_uS = cell(length(net_size)-1, 1);
            obj.forward_weights = cell(length(net_size)-1, 1);

            % Initialization: weight and bias

            % Initalize weight (non-software only)
            %if ~obj.software
            %    obj.array_interface.initialize_weights;
            %end

            for i=1:(length(net_size)-1)

                % Read initialized weights
                if ~obj.software
                    %obj.weights{i} = obj.array_interface.read_weights(i);
                else
                    obj.weights{i} = 1*(rand(net_size(i+1),net_size(i))-0.5);
                end

                if obj.software && obj.conductance_mode
                    % Differential pair: W = (G+ - G-) / k.
                    % Both physical conductances are in microsiemens.
                    k = (obj.G_max_uS - obj.G_min_uS) / obj.W_max;
                    Gmid = 0.5 * (obj.G_min_uS + obj.G_max_uS);
                    W0 = obj.weights{i};
                    obj.G_pos_uS{i} = Gmid + 0.5 * k * W0;
                    obj.G_neg_uS{i} = Gmid - 0.5 * k * W0;

                    % One independently sampled, fixed offset per device.
                    % Read noise is sampled later during each forward pass.
                    obj.noise_pos_uS{i} = zeros(size(W0));
                    obj.noise_neg_uS{i} = zeros(size(W0));
                    if strcmp(obj.noise_mode, 'static') && obj.G_noise_uS > 0
                        obj.noise_pos_uS{i} = obj.G_noise_uS * randn(size(W0));
                        obj.noise_neg_uS{i} = obj.G_noise_uS * randn(size(W0));
                    end
                end
                obj.forward_weights{i} = obj.weights{i};

                % Bias initialization (multiple methods here)
                obj.biases{i} = 1*(rand(net_size(i+1), 1)-0.5);

                % Momentum history initialization
                obj.last_nabla_w{i} = 0;
                obj.last_nabla_b{i} = 0;

                % RMSprop optimizer initialization
                if strcmp(obj.optimizer, 'RMSprop')
                    obj.grad_mean_sqr_w{i} = 0;
                    obj.grad_mean_sqr_b{i} = 0;
                end
            end
        end
        %%
        function output = forwardpass(obj, input, length_DPE_data_keep)
            % Forward pass
            % input: cell column vector

            % Convert cell vector to matrix
            input_matrix = cell2mat(input');

            for i=1:(length(obj.net_size)-1)

                % Matrix-vector multiplications
                if obj.software
                    W_eff = obj.weights{i};

                    if obj.conductance_mode
                        Gmin = obj.G_min_uS;
                        Gmax = obj.G_max_uS;
                        k = (Gmax - Gmin) / obj.W_max;

                        Gp = obj.G_pos_uS{i};
                        Gn = obj.G_neg_uS{i};

                        % Optional ideal 8-bit (or n-bit) conductance readout.
                        % Quantization precedes analogue conductance noise.
                        % The underlying trainable conductances stay continuous.
                        if isfinite(obj.conductance_bits)
                            step_uS = (Gmax - Gmin) / ...
                                (2^obj.conductance_bits - 1);
                            Gp = Gmin + step_uS * round((Gp - Gmin) / step_uS);
                            Gn = Gmin + step_uS * round((Gn - Gmin) / step_uS);
                        end

                        if strcmp(obj.noise_mode, 'static')
                            Gp = Gp + obj.noise_pos_uS{i};
                            Gn = Gn + obj.noise_neg_uS{i};
                        elseif obj.G_noise_uS > 0
                            % One noise realization per device and forward pass.
                            Gp = Gp + obj.G_noise_uS * randn(size(Gp));
                            Gn = Gn + obj.G_noise_uS * randn(size(Gn));
                        end

                        % No device may read outside the physical range.
                        Gp = min(Gmax, max(Gmin, Gp));
                        Gn = min(Gmax, max(Gmin, Gn));

                        W_eff = (Gp - Gn) / k;
                    end

                    % Cache effective weights for backpropagation.
                    obj.forward_weights{i} = W_eff;
                    temp_dpe = W_eff * input_matrix;
                else % array obj
                    obj.forward_weights{i} = obj.weights{i};
                    temp_dpe = obj.array_interface.dot_product(i, input_matrix);
                end

                temp_z = temp_dpe + repmat(obj.biases{i}, 1, length(input));

                % Keep results for backpropagation
                obj.z{i}=temp_z(:, 1:length_DPE_data_keep);

                % Activation
                if i < (length(obj.net_size) - 1)
                    temp_a = obj.relu(temp_z); % ReLU
                    input_matrix = temp_a; % Next input = activation now
                else
                    temp_a = temp_z; % Linear (last layer)
                    output = temp_a; % Output
                end

                % Keep those DPE for minibatch
                obj.a{i} = temp_a(:, 1:length_DPE_data_keep);
            end
        end
        %%
        function MSE_loss = update_mini_batch(obj, input, label)
            % Update weights/biases based on a minibatch

            % Initalize the variables to accumulate gradients
            for i = 1:(length(obj.net_size) - 1)
                obj.nabla_b{i} = zeros(size(obj.biases{i}));
                obj.nabla_w{i} = zeros(size(obj.weights{i}));
            end

            % Minibatch size
            [~, size_minibatch] = size(input);

            % Initialize loss
            MSE_loss = 0;

            % Backpropagate over minibatch
            for i = 1:size_minibatch
                % Backpropagation
                MSE_loss = MSE_loss + obj.backpropagation(input(:, i), label(:, i), i);
            end
            MSE_loss = MSE_loss / 2 / size_minibatch;

            % Update weight and biase
            for i = 1:(length(obj.net_size) - 1)

                % Devide by minibatch size
                obj.nabla_w{i} = 1 / size_minibatch * obj.nabla_w{i};
                obj.nabla_b{i} = 1 / size_minibatch * obj.nabla_b{i};

                if strcmp(obj.optimizer,'SGD')
                    % SGD optimizer
                    obj.nabla_w{i} = -obj.learningrate * obj.nabla_w{i} + ...
                        obj.momentum * obj.last_nabla_w{i};
                    obj.nabla_b{i} = -obj.learningrate * obj.nabla_b{i} + ...
                        obj.momentum * obj.last_nabla_b{i};
                elseif strcmp(obj.optimizer,'RMSprop')
                    % RMSprop optimizer
                    obj.grad_mean_sqr_w{i} = obj.decay * obj.grad_mean_sqr_w{i}...
                        + (1 - obj.decay) * obj.nabla_w{i}.^2;
                    obj.grad_mean_sqr_b{i} = obj.decay * obj.grad_mean_sqr_b{i}...
                        + (1 - obj.decay) * obj.nabla_b{i}.^2;
                    obj.nabla_w{i} = -obj.learningrate * obj.nabla_w{i}./(obj.grad_mean_sqr_w{i}.^0.5 + obj.eps)...
                        + obj.momentum * obj.last_nabla_w{i};
                    obj.nabla_b{i} = -obj.learningrate * obj.nabla_b{i}./(obj.grad_mean_sqr_b{i}.^0.5 + obj.eps)...
                        + obj.momentum * obj.last_nabla_b{i};
                else
                    error('Optimizer not supported yet.');
                end

                % ---------------------------------------------------------
                % Note: A trick to overcome the possible divergence at the
                % onset of the training with physical memristors, but
                % degrades performance
                % obj.nabla_w{i} = obj.nabla_w{i} - mean(obj.nabla_w{i}(:));
                % obj.nabla_b{i} = obj.nabla_b{i} - mean(obj.nabla_b{i}(:));
                % ---------------------------------------------------------

                % Save the recent nabla_w and nabla_b
                obj.last_nabla_w{i} = obj.nabla_w{i};
                obj.last_nabla_b{i} = obj.nabla_b{i};

                % Update biases with bounds (Note the bounds helps training
                % with physical memristors only)
                temp = obj.biases{i} + obj.nabla_b{i};
                temp(temp > 2.5) = 2.5; % Upper limit
                temp(temp < -2.5) = -2.5; % Lower limit
                obj.biases{i} = temp;

                % Software backend weight update
                if obj.software
                    if obj.conductance_mode
                        % Train G+ and G- in uS, not independent ideal W.
                        % A symmetric differential update produces delta_W.
                        k = (obj.G_max_uS - obj.G_min_uS) / obj.W_max;
                        dG_uS = 0.5 * k * obj.nabla_w{i};

                        obj.G_pos_uS{i} = min(obj.G_max_uS, ...
                            max(obj.G_min_uS, obj.G_pos_uS{i} + dG_uS));
                        obj.G_neg_uS{i} = min(obj.G_max_uS, ...
                            max(obj.G_min_uS, obj.G_neg_uS{i} - dG_uS));

                        % Nominal weights are derived from programmed G.
                        % The optional n-bit readout is applied in
                        % forwardpass, not to these stored target values.
                        obj.weights{i} = (obj.G_pos_uS{i} - ...
                            obj.G_neg_uS{i}) / k;
                    else
                        obj.weights{i} = obj.weights{i} + obj.nabla_w{i};
                    end
                end
            end

            % Hardware weight update
            if ~obj.software
                % Phyiscal update memristors of all layers together
                obj.array_interface.set_weights_gradient(obj.nabla_w);

                % Read new physical weights
                for i = 1:(length(obj.net_size) - 1)
                    obj.weights{i} = obj.array_interface.read_weights(i);
                end
            end
        end
        %%
        function single_loss=backpropagation(obj, input_single, label_single, sample_index)
            % Backpropagation of a single sample

            % Fetch forward pass output
            output_single = obj.a{length(obj.net_size) - 1}(:, sample_index);

            % Single sample loss
            single_loss = norm(label_single - output_single, 2);

            % Backward pass
            for i = (length(obj.net_size) - 1):-1:1

                % dE/dZ of current layer
                if i == (length(obj.net_size)-1)
                    % Last layer: linear activation
                    dErr_dz = obj.cost_derivative(output_single, label_single);
                else
                    % Rest layers: ReLU
                    % Backpropagate through the effective forward weights.
                    % Approximation: do not differentiate through clipping
                    % or (optional) conductance quantization.
                    dErr_dz = (obj.forward_weights{i+1}' * dErr_dz).*...
                        obj.relu_derivative(obj.z{i}(:, sample_index));
                end

                % dErr/db of current layer
                dErr_db = dErr_dz;

                % dErr/dw of current layer
                if i > 1
                    dErr_dw = dErr_dz * obj.a{i - 1}(:, sample_index)'; % Rest layer
                else
                    dErr_dw = dErr_dz * input_single'; % First layer
                end

                % Accumulate dE/db and dE/dw over minibatch
                obj.nabla_b{i} = obj.nabla_b{i} + dErr_db;
                obj.nabla_w{i} = obj.nabla_w{i} + dErr_dw;

            end
        end
    end
    methods(Static)
        %%
        function costderi_output=cost_derivative(output,label)
            % Delta of the output neuron (dE/d(output))
            % MSE error assumed
            % Output/label/costderi_output: a column vector

            costderi_output = -(label - output);
        end
        %%
        function relu_output=relu(relu_inputs)
            % ReLU function
            % relu_inputs/relu_outputs: matrix (each column one input)

            relu_output=relu_inputs.*(relu_inputs>0);
        end
        %%
        function reluderi_output = relu_derivative(reluderi_input)
            % ReLU derivative
            % reluderi_input/reluderi_output: a column vector

            reluderi_output = reluderi_input>0;
        end
    end
end
