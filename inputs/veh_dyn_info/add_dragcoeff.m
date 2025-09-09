% 读取CSV文件
data = csvread('all_ax_vs_speed_poly.csv', 1); % 假设第一行是标题

% 提取v和ax列
v_original = data(:, 1);   % 原始速度数据
ax_original = data(:, 2);  % 原始ax数据

% 计算修正值并修改ax（原始数据）
% correction_original = (-(-1.04765 * v_original.^2 + 47.5 * v_original - 1609.73)) / 760;
correction_original = (1.194 * v_original.^2 ) / 760;
modified_ax_original = ax_original + correction_original;

% 生成70-80的速度点（间隔0.1，可根据需要调整）
v_extend = 70:0.1:80;

% 使用原始数据拟合多项式（这里使用3次多项式，可根据拟合效果调整次数）
% 只使用原始数据中v<=70的部分进行拟合
valid_idx = v_original <= 70;
p = polyfit(v_original(valid_idx), modified_ax_original(valid_idx), 3);

% 用拟合的多项式计算70-80的ax值
ax_extend = polyval(p, v_extend);

% 计算扩展部分的修正值
correction_extend = (-(-1.04765 * v_extend.^2 + 47.5 * v_extend - 1609.73)) / 760;

% 应用修正
modified_ax_extend = ax_extend + correction_extend;

% 合并原始数据和扩展数据
v_combined = [v_original; v_extend'];
ax_combined = [modified_ax_original; modified_ax_extend'];

% 组合修改后的数据
modified_data = [v_combined, ax_combined];

% 保存到新文件
csvwrite('all_ax_vs_speed_poly_adddrag.csv', modified_data);

% 显示完成信息
fprintf('数据处理完成，已保存到 all_ax_vs_speed_poly_adddrag.csv\n');
fprintf('原始数据范围: %.1f - %.1f\n', min(v_original), max(v_original));
fprintf('扩展后数据范围: %.1f - %.1f\n', min(v_combined), max(v_combined));
    