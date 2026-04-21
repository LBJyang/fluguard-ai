# 流感风险评估与早期预警 | Flu Risk Assessment and Early Warning

## 多维度风险评估框架 | Multi-Dimensional Risk Framework

FluGuard 流感卫士系统整合以下维度进行风险评估：

### 1. 咳嗽监测数据 (Cough Detection — 30%)

YAMNet迁移学习模型实时识别教室内咳嗽声音，提供定量预警信号：

| 24小时咳嗽次数/班 | 风险等级 | 建议行动 |
|---|---|---|
| <30 | 低 LOW | 常规监测 |
| 30–80 | 中低 MEDIUM-LOW | 加强通风，关注趋势 |
| 80–150 | 中 MEDIUM | 强化晨检，通知家长 |
| 150–250 | 高 HIGH | 全面评估，上报卫生室 |
| >250 | 极高 VERY HIGH | 停课建议，紧急消毒 |

Cough count thresholds: <30/24h = Low; 30–80 = Medium-Low; 80–150 = Medium; 150–250 = High; >250 = Very High/Consider suspension.

### 2. 环境传感器数据 (Environmental — 25%)

**温湿度综合风险指数:**
- 温度<5°C + 湿度<40%：病毒存活时间最长，传播风险极高（指数3.0）
- 温度5-10°C + 湿度40-55%：传播风险高（指数2.0）
- 温度10-15°C + 湿度55-65%：传播风险中（指数1.5）
- 温度>15°C 或 湿度>65%：传播风险低（指数1.0）

**室内空气质量:**
- CO2 <800ppm：通风良好，低风险
- CO2 800-1200ppm：通风一般，建议开窗
- CO2 >1200ppm：通风不足，须立即改善

Temperature <5°C with humidity <40% maximizes virus survival (index 3.0). Indoor CO2 >1200 ppm indicates poor ventilation — immediate action required.

### 3. 社区流行病学数据 (Community Epidemiology — 25%)

- 周边医院发热门诊流感样病例数（同期比较）
- 当地CDC流感哨点监测数据
- 本校历史同期发病率对比

Hospital ED flu patient counts, local CDC sentinel surveillance data, and historical school-level flu rates for same period in prior years.

### 4. 学生健康打卡数据 (Health Check-in — 20%)

通过App收集学生主动上报数据：
- 班级出勤率（缺勤增加是早期信号）
- 发烧上报率
- 症状自评分布

Student app check-in data: absenteeism rates (early warning signal), fever reports, and symptom self-assessment distribution.

## 综合风险评分算法 | Composite Risk Score Algorithm

```
风险评分(0-100) = 
  咳嗽指数(0-40) × 0.30 +
  环境指数(0-40) × 0.25 +
  社区指数(0-40) × 0.25 +
  打卡指数(0-40) × 0.20
```

**评分解读:**
- 0-25：绿色（低风险）—— 常规防控
- 26-50：黄色（中等风险）—— 加强监测，通知家长
- 51-75：橙色（高风险）—— 启动应急预案，报告上级
- 76-100：红色（极高风险）—— 考虑停课，向疾控报告

Score interpretation: 0–25 (Green, Low), 26–50 (Yellow, Moderate), 51–75 (Orange, High), 76–100 (Red, Very High — consider school closure and CDC reporting).

## 早期预警信号识别 | Early Warning Signal Recognition

以下信号组合出现时，应提前行动：

**信号组合1：咳嗽趋势上升 + 气温骤降**
- 连续2-3天咳嗽计数增长>20%
- 气温在3天内下降超过8°C
- → 提前发送家长提醒，加强班级通风

**信号组合2：多班级同时出现症状**
- ≥3个班级同日出现发烧学生
- → 可能为全校爆发，立即向学校卫生室报告

**信号组合3：社区高压 + 教室高CO2**
- 医院发热门诊流感患者超出平均值50%
- 教室CO2 > 1200ppm
- → 增加通风频率，考虑错峰上课

Early warning combinations: (1) Rising cough trend + sudden temperature drop → parent alerts + enhanced ventilation; (2) ≥3 classes with fever cases same day → possible school-wide outbreak, report immediately; (3) Hospital flu surge + high CO2 → increase ventilation frequency.

## 历史数据参考：中国北方校园流感规律 | Historical Patterns

根据中国疾控中心数据（2018-2023年综合分析）：

- 北方学校流感峰值通常在1月（寒假前）和3月（开学后）
- 小学生（6-12岁）年流感发病率约为30-40%
- 班级内传播：指数期（第1周）传播至约30%同学
- 停课干预：及时停课可减少约40-60%的校内传播

Northern China school flu peaks: January (before winter break) and March (after school resumes). Incidence in primary school students: 30–40% annually. Class outbreak dynamics: ~30% of classmates infected in week 1. Timely class suspension reduces school transmission by 40–60%.

## 各角色决策参考 | Role-Specific Decision Reference

**教育局 Education Bureau:**
- 触发区域预警阈值：辖区内>15%的学校出现高风险评分
- 考虑区域停课：>3所学校同期发生聚集疫情
- 启动应急预案：联系疾控中心，协调医疗资源

**校长 Principal:**
- 启动学校应急预案：全校风险评分>60
- 联系区疾控报告：全校流感样病例率>5%
- 考虑停课：全校流感样病例率>10% 或 出现重症病例

**班主任 Teacher:**
- 通知家长接回学生：当日发烧或明显症状
- 加强消毒：班级风险等级升至中等
- 向卫生室报告：当日班内≥3名学生发烧

**家长 Parent:**
- 体温≥37.5°C不送学
- 接到学校通知后2小时内接回学生
- 居家隔离满足返校标准后再送学

Bureau triggers regional action when >15% of schools show high risk. Principals initiate emergency response at school-wide risk score >60. Teachers notify parents when any student has fever. Parents should keep feverish children home (≥37.5°C).
