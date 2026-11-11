# 超市销售：销售额升、毛利润降 —— 诊断项目

## 问题

模拟某超市 12 个月销售明细：账面**销售额上升**，但**毛利润与毛利率下降**。  
用数据分析 + 多种机器学习模型找原因 / 做预警。

## 用了哪些算法？

| 脚本 | 算法 | 任务类型 | 预测什么 |
|------|------|----------|----------|
| `diagnose_margin.py` | **随机森林回归** `RandomForestRegressor` | 回归 | 连续「毛利率」+ 特征重要性 |
| `classify_margin_logreg.py` | **逻辑回归** `LogisticRegression` | 二分类 | 是否「低毛利风险单」 |
| `classify_margin_svm.py` | **SVM** `LinearSVC` / `SVC(rbf)` | 二分类 | 同上 |

分类标签定义：`毛利率 < 15%` → 低毛利(1)，否则正常(0)。

## Mock 数据里故意埋的 3 个坑

1. **品类结构恶化**：低毛利「促销特价」销量暴增，高毛利「生鲜」占比下降  
2. **折扣加深**：整体折扣率爬升，团购更深  
3. **成本上涨、售价僵硬**：进货成本抬升，标价几乎不动

## 运行

在仓库根目录：

```powershell
.\.venv\Scripts\Activate.ps1
pip install openpyxl

# 生成 Excel（可省略：下面脚本会自动生成）
python projects\sales_margin_diagnosis\mock_sales_excel.py

# 1) 随机森林回归：解释连续毛利率
python projects\sales_margin_diagnosis\diagnose_margin.py

# 2) 逻辑回归：低毛利订单分类 + 系数解释
python projects\sales_margin_diagnosis\classify_margin_logreg.py

# 3) SVM：线性核 + RBF 核对比
python projects\sales_margin_diagnosis\classify_margin_svm.py
```

## 产出

| 路径 | 说明 |
|------|------|
| `data/supermarket_sales.xlsx` | mock 销售明细 |
| `output/01_monthly_trends.png` | 销售额↑ 毛利↓ |
| `output/02_category_mix.png` | 品类结构 |
| `output/03_feature_importance.png` | 随机森林置换重要性 |
| `output/04_logreg_confusion.png` | 逻辑回归混淆矩阵 |
| `output/05_logreg_coefficients.png` | 逻辑回归系数 |
| `output/06_svm_confusion.png` | SVM 混淆矩阵 |
| `output/07_svm_rbf_importance.png` | RBF-SVM 重要性 |

## 怎么理解三者关系

1. **随机森林回归**：回答「毛利率高低主要由哪些因素驱动」  
2. **逻辑回归 / SVM 分类**：回答「这笔订单像不像低毛利风险单」，可做业务预警  
3. **逻辑回归 / 线性 SVM**：系数可解释；**RBF SVM**：边界更灵活，用置换重要性辅助解释
