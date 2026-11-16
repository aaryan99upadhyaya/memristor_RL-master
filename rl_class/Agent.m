classdef Agent < handle
    % Learning agent
    
    properties
        
        memory_capacity = 20000; % No. of experiences to hold        
        batch_size=80; % Minibatch size
        replay_start_size = 400; % Kind of like warm start
        
        % Discount factor
        gamma = 0.99;
        
        % Greedy policy parameters
        max_epsilon = 1;
        min_epsilon = 0.01;
        lambda = 0.005; % speed of decay
        
        % Input and Output size of Q net
        stateCnt
        actionCnt

        % Interface with other classes
        brain        % Online network  - trained every step
        brain_target % Target network - frozen, copied from brain periodically
        memory

        % Target-network update frequency (steps)
        target_update_freq = 500;

        % Exploration probability
        epsilon
        steps % Truly total steps, determining epsilon

        % Store the forward pass of stat_, for action prediction
        stat_predict

    end

    methods
        %%
        function obj = Agent(net_size, varargin)
            % Create the agent obj

            % Input (state vector) size
            obj.stateCnt = net_size(1);

            % Output (SET of action) size
            obj.actionCnt = net_size(end);

            % Online network
            obj.brain = Brain(net_size, varargin{:});

            % Target network: same architecture, weights copied from brain
            obj.brain_target = Brain(net_size, varargin{:});
            obj.brain_target.copy_weights_from(obj.brain);

            obj.memory = Memory(obj.memory_capacity);

            % Initial episilon
            obj.epsilon = obj.max_epsilon;
            obj.steps = 0;

            % The initial Q-network forward pass result of stat (nothing)
            obj.stat_predict = NaN;
        end
        %%
        function action = act(obj)
            % Suggesting an action based on current state

            % Random explore (probability, or no prediction available round)
            if or(rand < obj.epsilon, isnan(obj.stat_predict))
                % Random exploration
                action = -1 + 2 * (rand > 0.5); % Either -1 or 1
            else
                % Greedy policy from Q-network
                [~,action] = max(obj.stat_predict);
                if action == 1
                    action = -1; % N1 win: action -1
                else
                    action = 1; % N2 win: action 1
                end
            end
        end
        %%
        function observe(obj, new_sample)
            % Add observation to memory & change exploration probability            
            
            % Add {state, action, reward, next_state} to memory
            obj.memory.add_memory(new_sample);

            % The time dependent probability of exploration
            obj.steps = obj.steps + 1;
            obj.epsilon = obj.min_epsilon + (obj.max_epsilon - obj.min_epsilon) * exp(-obj.lambda * obj.steps);

            % Periodically copy online weights to target network
            if mod(obj.steps, obj.target_update_freq) == 0
                obj.brain_target.copy_weights_from(obj.brain);
            end
        end
        %%
        function MSE_loss = replay_predict(obj, stat_to_pred)
            % (1) Replay and learn from experience using a frozen target network
            % (2) Q-network forwardpass for action prediction (stat_to_pred)

            % Fetch a batch of samples from memory
            minibatch = obj.memory.sample_memory(obj.batch_size);

            % Verify the batch size (it may < obj.batch_size)
            batchlen = size(minibatch, 1);

            % States and next_states of the minibatch
            states = minibatch(1:batchlen, 1);
            % Next states (of all samples in the minibatch)
            states_ = minibatch(1:batchlen, 4);

            % Wrap stat_to_pred (4x1 double) to a cell
            stat_to_pred_cell{1,1} = stat_to_pred;

            % --- Online network forward pass ---
            % Run only current states + stat_to_pred through the online net.
            temp_online = obj.brain.predict([states; stat_to_pred_cell], length(states));
            p = temp_online(:, 1:length(states));
            obj.stat_predict = temp_online(:, end);

            % --- Target network forward pass ---
            % Run next states through the FROZEN target network to compute
            % Bellman targets.
            p_ = obj.brain_target.predict(states_, 0);

            % Prepare the batch 'Inputs' and 'Labels' for training
            train_input = zeros(obj.stateCnt, batchlen);
            train_label = zeros(obj.actionCnt, batchlen);

            for i = 1:batchlen

                s = minibatch{i, 1}; % Get state
                a = minibatch{i, 2}; % action
                r = minibatch{i, 3}; % reward
                s_ = minibatch{i, 4}; % next_state

                % Initial 'label' with current inference of (s), 2x1 vector
                t = p(:, i);

                % Bellman equation (target from frozen network)
                if isnan(s_)
                    % Terminal step: label is just the immediate reward
                    t(1 + (a > 0)) = r;
                else
                    % Non-terminal: r + gamma * max Q_target(s', .)
                    t(1 + (a > 0)) = r + obj.gamma * max(p_(:, i));
                end

                % Assign "Input" and "Label"
                train_input(:, i) = s;
                train_label(:, i) = t;
            end

            % Train the online network
            MSE_loss = obj.brain.train(train_input, train_label);
        end
    end
end
