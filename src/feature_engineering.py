import numpy as np
import pandas as pd


def apply_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()

    # 1. Payment delay features
    pay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

    df_fe['avg_delay'] = df_fe[pay_cols].mean(axis=1)
    df_fe['max_delay'] = df_fe[pay_cols].max(axis=1)
    df_fe['min_delay'] = df_fe[pay_cols].min(axis=1)
    df_fe['num_delays'] = (df_fe[pay_cols] > 0).sum(axis=1)
    df_fe['num_severe_delays'] = (df_fe[pay_cols] >= 2).sum(axis=1)

    # 2. Bill amount behavior
    bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']

    df_fe['avg_bill'] = df_fe[bill_cols].mean(axis=1)
    df_fe['max_bill'] = df_fe[bill_cols].max(axis=1)
    df_fe['min_bill'] = df_fe[bill_cols].min(axis=1)
    df_fe['bill_std'] = df_fe[bill_cols].std(axis=1)
    df_fe['bill_growth'] = df_fe['BILL_AMT6'] - df_fe['BILL_AMT1']

    # bill_trend using polyfit (same as notebook)
    bill_matrix = df_fe[bill_cols].values
    df_fe['bill_trend'] = np.polyfit(range(6), bill_matrix.T, 1)[0]

    # 3. Payment amount behavior
    pay_amt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

    df_fe['avg_pay_amt'] = df_fe[pay_amt_cols].mean(axis=1)
    df_fe['max_pay_amt'] = df_fe[pay_amt_cols].max(axis=1)
    df_fe['min_pay_amt'] = df_fe[pay_amt_cols].min(axis=1)
    df_fe['pay_amt_std'] = df_fe[pay_amt_cols].std(axis=1)
    df_fe['pay_amt_growth'] = df_fe['PAY_AMT6'] - df_fe['PAY_AMT1']

    # 4. Ratios
    df_fe['utilization_ratio'] = df_fe['avg_bill'] / (df_fe['LIMIT_BAL'] + 1)
    df_fe['repayment_ratio'] = df_fe['avg_pay_amt'] / (df_fe['avg_bill'] + 1)
    df_fe['income_to_limit'] = df_fe['LIMIT_BAL'] / (df_fe['BILL_AMT1'] + 1)

    # 5. Volatility
    df_fe['bill_volatility'] = df_fe[bill_cols].std(axis=1)
    df_fe['payment_volatility'] = df_fe[pay_amt_cols].std(axis=1)

    # 6. Debt acceleration
    df_fe['debt_acceleration'] = \
        (df_fe['BILL_AMT6'] - df_fe['BILL_AMT5']) - \
        (df_fe['BILL_AMT2'] - df_fe['BILL_AMT1'])

    # 7. Total features
    df_fe['total_bill_6m'] = df_fe[bill_cols].sum(axis=1)
    df_fe['total_pay_6m'] = df_fe[pay_amt_cols].sum(axis=1)
    df_fe['difference_bill_pay'] = df_fe['total_bill_6m'] - df_fe['total_pay_6m']

    return df_fe

