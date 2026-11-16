classdef Memory < handle
    % Memory (of the agent)

    properties
        capacity
        samples
        pointer = 0; % Current write position in the ring array
        count   = 0; % Number of valid entries max of cspacity
    end

    methods
        %%
        function obj = Memory(capacity)
            % Create memory obj

            % Size of the memory
            obj.capacity = capacity;

            % Initial memory with the given size
            obj.samples = cell(capacity, 4);
        end
        %%
        function add_memory(obj, new_sample)
            % Add {state, action, reward, next_state} 1x4 cell to memory.
            % using mod like a circular changing to keep the same capacity
            obj.pointer = mod(obj.pointer, obj.capacity) + 1;
            obj.samples(obj.pointer, 1:4) = new_sample;
            if obj.count < obj.capacity
                obj.count = obj.count + 1;
            end
        end
        %%
        function samples_recalled = sample_memory(obj, n)
            % Sample the experience

            % Draw at most as many samples as we have stored
            n = min(n, obj.count);

            % Recall n different samples from the existing memory
            samples_index = sort(datasample(1:obj.count, n, 'Replace', false));

            % The recalled experience, cell array nx4 size.
            samples_recalled = obj.samples(samples_index, 1:4);
        end
    end
end
