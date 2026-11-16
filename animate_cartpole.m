%% 
%% CART-POLE EPISODE REPLAY

clear;
clc;
close all;

% Adding these elements to the filepath
addpath(genpath(fullfile(pwd, 'game_simulator_class')))
addpath(genpath(fullfile(pwd, 'xbar_backend_class')))
addpath(genpath(fullfile(pwd, 'rl_class')))
addpath(genpath(fullfile(pwd, 'dense_network_class')))
addpath(genpath(fullfile(pwd, 'array_class')))

% Load training data
load('cartpole_training_runs.mat');

% -----------------------------------------
% SELECT WHICH EPISODE TO ANIMATE
% -----------------------------------------
%episode_to_simulate = 110; % only the final, best episode is saved

% Check that episode exists
%if episode_to_simulate < 1 || ...
%        episode_to_simulate > length(episodeData)
%
%    error('Invalid episode number.');
%
%end

% Retrieve selected episode
%run = episodeData{episode_to_simulate};
run = episodeData{1};

% Create a new cart-pole object
env = CartPole();

% -----------------------------------------
% CREATE ANIMATION AND LIVE DATA PLOTS
% -----------------------------------------

numSteps = size(run.states,1);

% Create a tiled figure
figure('Name','Cart-Pole Simulation','Color','w');

t = tiledlayout(2,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');

% Animation occupies the left side
ax_anim = nexttile([2 1]);

% Position plot
ax_pos = nexttile;

% Angle plot
ax_angle = nexttile;

% -----------------------------------------
% INITIALIZE LIVE PLOTS
% -----------------------------------------

% Cart position
h_pos = animatedline(ax_pos, ...
    'LineWidth',1.5);

xlabel(ax_pos,'Time (s)');
ylabel(ax_pos,'Position (m)');
title(ax_pos,'Cart Position');
grid(ax_pos,'on');

xlim(ax_pos,[run.times(1),run.times(end)]);
ylim(ax_pos,[-2.5 2.5]);

% Pole angle
h_angle = animatedline(ax_angle, ...
    'LineWidth',1.5);

xlabel(ax_angle,'Time (s)');
ylabel(ax_angle,'Angle (degrees)');
title(ax_angle,'Pole Angle');
grid(ax_angle,'on');

xlim(ax_angle,[run.times(1),run.times(end)]);
ylim(ax_angle,[-15 15]);

% -----------------------------------------
% ANIMATE CART-POLE WITH LIVE LABELS
% -----------------------------------------

numSteps = size(run.states,1);

% Create figure
figure('Name','Cart-Pole Neural Network Simulation',...
    'Color','w');

ax = axes;
env = CartPole();

% Create live data label
h_data = text(ax, 0, 0, '', ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'BackgroundColor','white', ...
    'EdgeColor','black', ...
    'Margin',7, ...
    'VerticalAlignment','middle', ...
    'HorizontalAlignment','left');

% -----------------------------------------
% ANIMATION LOOP
% -----------------------------------------

for k = 1:numSteps

    % Retrieve recorded state
    cart_position = run.states(k,1);
    pole_angle = run.states(k,3);

    % Convert pole angle to degrees
    pole_angle_deg = pole_angle*180/pi;

    % Update cart-pole animation
    env.state_now = run.states(k,:)';
    env.CartPole_plot(ax,k-1);

    % Position live label next to the pole
    % Keep label to the right of the cart
    label_x = cart_position + 0.65;
    label_y = 0.9;

    % Update numerical label
    set(h_data, ...
        'Position',[label_x label_y 0], ...
        'String',sprintf( ...
        'CART POSITION\nx = %.3f m\n\nPOLE ANGLE\n\\theta = %.2f deg', ...
        cart_position, ...
        pole_angle_deg));

    % Update title
    title(ax,sprintf('Best episode | Time = %.2f s', ...
        run.times(k)));

    % Refresh animation
    drawnow limitrate;

    % Match original simulation timing
    if k < numSteps
        pause(run.times(k+1)-run.times(k));
    end

end

%% PLOT SIMULATION DATA

figure('Name','Cart-Pole Simulation Data');

subplot(3,1,1);

plot(run.times,run.states(:,1),'LineWidth',1.5);

xlabel('Time (s)');
ylabel('Position (m)');
title('Cart Position');
grid on;

subplot(3,1,2);

plot(run.times,run.states(:,3)*180/pi,'LineWidth',1.5);

xlabel('Time (s)');
ylabel('Angle (degrees)');
title('Pole Angle');
grid on;

subplot(3,1,3);

stairs(run.times(1:end-1),run.actions,'LineWidth',1.5);

xlabel('Time (s)');
ylabel('Action');
title('Neural Network Control Output');
yticks([-1 1]);
grid on;