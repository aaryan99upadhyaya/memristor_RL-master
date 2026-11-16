classdef Brain < handle
    % Brain (of the agent)

    properties

        % Size of network
        net_size

        % Neural network
        net
    end

    methods
        %%
        function obj = Brain(net_size, varargin)
            % Create brain obj

            obj.net_size = net_size;
            
            % Creat a two-layer feedforward fully connected network
            obj.net = arblayerfullconn(net_size, varargin{:});

        end
        %%
        function o = predict(obj, stat, length_DPE_data_keep)
            % Fowwardpass of Q-net

            % Flags of NaN cells
            flag_NaN = cellfun(@(x) any(isnan(x(:))), stat); % Column logic vector
            % NaN free input to DPE
            input = stat(~flag_NaN);

            % Inference return (each column one inference)
            temp = obj.net.forwardpass(input, length_DPE_data_keep);

            % Prepare output matrix by including NaNs back
            o = NaN(obj.net_size(end), length(stat));
            flag_NaN_matrix = repmat(flag_NaN', obj.net_size(end), 1);
            o(~flag_NaN_matrix) = temp;
        end
        %%
        function MSE_loss = train(obj, input, label)
            % Train the Q-net

            MSE_loss = obj.net.update_mini_batch(input, label);
        end
        %%
        function copy_weights_from(obj, src_brain)
            % Copy weights, biases, and conductance arrays from src_brain
            n_layers = length(src_brain.net.weights);
            for i = 1:n_layers
                obj.net.weights{i} = src_brain.net.weights{i};
                obj.net.biases{i}  = src_brain.net.biases{i};
            end
            % Sync conductance-mode arrays onl when in use
            if ~isempty(src_brain.net.G_pos_uS) && ~isempty(src_brain.net.G_pos_uS{1})
                obj.net.G_pos_uS = src_brain.net.G_pos_uS;
                obj.net.G_neg_uS = src_brain.net.G_neg_uS;
                obj.net.noise_pos_uS = src_brain.net.noise_pos_uS;
                obj.net.noise_neg_uS = src_brain.net.noise_neg_uS;
            end
        end
    end
end
