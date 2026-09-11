%% EMMA — Accuracy vs Compression Ratio Plot
% Colorblind-safe palette (Wong 2011), Times New Roman font

clear; close all;

%% Data
dims = [32, 64, 128];

acc_pca     = [89.1, 91.9, 90.6];
acc_ae      = [83.4, 84.7, 87.1];
acc_vae     = [84.1, 84.9, 84.0];
acc_distil  = [86.5, 87.9, 89.7];   % 32, 64, 128 dims
acc_nocomp  = 89.2;                  % horizontal reference

%% Colorblind-safe palette (Wong 2011)
c_pca    = [0,   114, 178] / 255;   % blue
c_ae     = [230, 159,   0] / 255;   % orange
c_vae    = [0,   158, 115] / 255;   % green
c_distil = [213,  94,   0] / 255;   % vermillion
c_nocomp = [0.4, 0.4, 0.4];         % grey

%% Figure
fig = figure('Units','inches','Position',[1 1 6 4]);

hold on;

% No compression reference line
yline(acc_nocomp, '--', 'Color', c_nocomp, 'LineWidth', 1.5, ...
      'Label', 'No compression (89.2%)', ...
      'LabelHorizontalAlignment', 'left', ...
      'FontName', 'Times New Roman', 'FontSize', 10);

% Model lines
plot(dims, acc_pca,    '-o', 'Color', c_pca,    'LineWidth', 2, ...
     'MarkerSize', 7, 'MarkerFaceColor', c_pca);
plot(dims, acc_ae,     '-s', 'Color', c_ae,     'LineWidth', 2, ...
     'MarkerSize', 7, 'MarkerFaceColor', c_ae);
plot(dims, acc_vae,    '-^', 'Color', c_vae,    'LineWidth', 2, ...
     'MarkerSize', 7, 'MarkerFaceColor', c_vae);
plot(dims, acc_distil, '-d', 'Color', c_distil, 'LineWidth', 2, ...
     'MarkerSize', 7, 'MarkerFaceColor', c_distil);

hold off;

%% Axes
set(gca, 'XTick', dims, 'XTickLabel', {'32 (16×)', '64 (8×)', '128 (4×)'}, ...
         'FontName', 'Times New Roman', 'FontSize', 11);

xlabel('Latent Dimension (Compression Ratio)', ...
       'FontName', 'Times New Roman', 'FontSize', 12);
ylabel('Test Accuracy (%)', ...
       'FontName', 'Times New Roman', 'FontSize', 12);
title('Accuracy vs. Compression Ratio on COCO Matching', ...
      'FontName', 'Times New Roman', 'FontSize', 13);

ylim([80, 94]);
xlim([24, 136]);
grid on;
box on;

%% Legend
legend({'No compression', 'PCA', 'AE', 'VAE', 'DistilAE'}, ...
       'Location', 'southwest', ...
       'FontName', 'Times New Roman', 'FontSize', 10);

%% Export
exportgraphics(fig, 'compression_accuracy.pdf', 'ContentType', 'vector');
exportgraphics(fig, 'compression_accuracy.png', 'Resolution', 300);

disp('Saved: compression_accuracy.pdf / .png');
