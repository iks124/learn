import numpy as np
from scipy import stats

# 数据
gnp = np.array([1, 2, 3, 5, 8])          # 国民生产总值（10亿美元）
poverty = np.array([11, 12, 13, 15, 18]) # 贫困比例（%）

# 皮尔逊相关分析
r, p_value = stats.pearsonr(gnp, poverty)

print(f"皮尔逊相关系数 r = {r:.4f}")
print(f"p-value = {p_value:.4f}")
