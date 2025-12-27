"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""



def create_submission(submission):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)

    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random
    
    import sklearn
    from sklearn.linear_model import Ridge
    
    import catboost
    from catboost import CatBoostRegressor, Pool

    RANDOM_SEED = 322
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train = pd.read_csv('train.csv', parse_dates=['dt'])
    test = pd.read_csv('test.csv',  parse_dates=['dt'])

    base_features = [
        'management_group_id',
        'first_category_id',
        'second_category_id',
        'third_category_id',
        'product_id',
        'dow',
        'day_of_month',
        'week_of_year',
        'month',
        'n_stores',
        'holiday_flag',
        'activity_flag',
        'precpt',
        'avg_temperature',
        'avg_humidity',
        'avg_wind_level'
    ]
    
    cat_features = [
        'management_group_id',
        'first_category_id',
        'second_category_id',
        'third_category_id',
        'product_id',
        'dow',
        'holiday_flag',
        'activity_flag'
    ]

    train = train.sort_values(['product_id', 'dt']).reset_index(drop=True)
    test  = test.sort_values(['product_id', 'dt']).reset_index(drop=True)

    for col in ['price_p05', 'price_p95']:
        train[f'{col}_lag1'] = train.groupby('product_id')[col].shift(1)
    
    train['price_width_lag1'] = train['price_p95_lag1'] - train['price_p05_lag1']
    
    for col in ['price_p05', 'price_p95']:
        train[f'{col}_lag7'] = train.groupby('product_id')[col].shift(7)
    
    train['p05_roll7'] = train.groupby('product_id')['price_p05'] \
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    
    train['p95_roll7'] = train.groupby('product_id')['price_p95'] \
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    
    train['price_width'] = train['price_p95'] - train['price_p05']
    
    train['width_roll7'] = train.groupby('product_id')['price_width'] \
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    
    train['width_std14'] = train.groupby('product_id')['price_width'] \
        .transform(lambda x: x.shift(1).rolling(14, min_periods=5).std())
    
    train['width_mean14'] = train.groupby('product_id')['price_width'] \
        .transform(lambda x: x.shift(1).rolling(14, min_periods=5).mean())
    
    train['width_cv14'] = train['width_std14'] / (train['width_mean14'] + 1e-6)

    lag_cols = [
        'price_p05_lag1', 'price_p95_lag1', 'price_width_lag1',
        'price_p05_lag7', 'price_p95_lag7',
        'p05_roll7', 'p95_roll7', 'width_roll7'
    ]
    lag_cols += ['width_std14', 'width_mean14', 'width_cv14']
    
    
    for col in lag_cols:
        train[col] = train.groupby('product_id')[col] \
            .transform(lambda x: x.fillna(x.expanding().mean()))
        train[col] = train[col].fillna(0.0)
    
    last_vals = train.groupby('product_id')[lag_cols].last().reset_index()
    
    test = test.merge(last_vals, on='product_id', how='left')
    
    for col in lag_cols:
        test[col] = test[col].fillna(last_vals[col].mean())
    
    features = base_features + lag_cols

    split_date = train['dt'].max() - pd.Timedelta(days=14)

    train_df = train[train['dt'] < split_date]
    val_df   = train[train['dt'] >= split_date]
    
    train_pool_p05 = Pool(
        train_df[features], train_df['price_p05'], cat_features=cat_features
    )
    val_pool_p05 = Pool(
        val_df[features], val_df['price_p05'], cat_features=cat_features
    )
    
    train_pool_p95 = Pool(
        train_df[features], train_df['price_p95'], cat_features=cat_features
    )
    val_pool_p95 = Pool(
        val_df[features], val_df['price_p95'], cat_features=cat_features
    )

    configs = [
        {'depth': 6,  'learning_rate': 0.03, 'iterations': 800},
        {'depth': 8,  'learning_rate': 0.02, 'iterations': 1000},
        {'depth': 10, 'learning_rate': 0.01, 'iterations': 1500},
    ]
    
    models_p05, models_p95 = [], []
    
    for cfg in configs:
        m05 = CatBoostRegressor(
            loss_function='Quantile:alpha=0.05',
            random_seed=RANDOM_SEED,
            early_stopping_rounds=50,
            verbose=0,
            **cfg
        ).fit(train_pool_p05, eval_set=val_pool_p05)
    
        m95 = CatBoostRegressor(
            loss_function='Quantile:alpha=0.95',
            random_seed=RANDOM_SEED,
            early_stopping_rounds=50,
            verbose=0,
            **cfg
        ).fit(train_pool_p95, eval_set=val_pool_p95)
    
        models_p05.append(m05)
        models_p95.append(m95)

    eps = 1e-6

    val_p05_preds = np.column_stack([m.predict(val_df[features]) for m in models_p05])
    val_p95_preds = np.column_stack([m.predict(val_df[features]) for m in models_p95])
    
    val_p05_ens = np.quantile(val_p05_preds, 0.7, axis=1)
    val_p95_ens = np.quantile(val_p95_preds, 0.3, axis=1)
    val_p95_ens = np.maximum(val_p05_ens + eps, val_p95_ens)
    
    prod_width = train_df.groupby('product_id')['price_width'].mean()
    narrow_thr = prod_width.quantile(0.3)
    
    narrow_products = set(prod_width[prod_width <= narrow_thr].index)
    
    val_df['is_narrow'] = val_df['product_id'].isin(narrow_products)
    test['is_narrow']   = test['product_id'].isin(narrow_products)

    def iou(l_t, u_t, l_p, u_p):
        inter = np.maximum(0, np.minimum(u_t, u_p) - np.maximum(l_t, l_p))
        union = (u_t - l_t) + (u_p - l_p) - inter
        return inter / (union + 1e-9)
    
    center = (val_p05_ens + val_p95_ens) / 2
    width  = val_p95_ens - val_p05_ens
    
    def optimal_c_for_row(l_t, u_t, center, width, c_grid=np.linspace(0.3, 1.2, 91)):
        best_c, best_iou = 1.0, -1
        for c in c_grid:
            l = center - width * c / 2
            u = center + width * c / 2
            iou_val = iou(np.array([l_t]), np.array([u_t]), l, u)
            if iou_val > best_iou:
                best_iou, best_c = iou_val, c
        return best_c

    val_df['center'] = center
    val_df['width']  = width
    val_df['optimal_c'] = [
        optimal_c_for_row(l, u, c, w)
        for l, u, c, w in zip(
            val_df['price_p05'], val_df['price_p95'],
            val_df['center'], val_df['width']
        )
    ]
    
    calib_features = ['width', 'width_cv14', 'n_stores', 'dow', 'holiday_flag', 'is_narrow']
    X_calib = val_df[calib_features].fillna(0).astype(float)
    y_calib = val_df['optimal_c']
    
    c_calibrator = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    c_calibrator.fit(X_calib, y_calib)
    
    c_val = c_calibrator.predict(X_calib)
    c_val = np.clip(c_val, 0.4, 1.2)  
    
    val_df['price_p05_pred'] = val_df['center'] - val_df['width'] * c_val / 2
    val_df['price_p95_pred'] = val_df['center'] + val_df['width'] * c_val / 2
    val_df['price_p95_pred'] = np.maximum(val_df['price_p05_pred'] + eps, val_df['price_p95_pred'])

    def iou_metric(l_true, u_true, l_pred, u_pred, eps=1e-9):
        inter = np.maximum(0, np.minimum(u_true, u_pred) - np.maximum(l_true, l_pred))
        union = (u_true - l_true) + (u_pred - l_pred) - inter
        return inter / (union + eps)
    
    val_iou = iou_metric(
        val_df['price_p05'].values,
        val_df['price_p95'].values,
        val_df['price_p05_pred'].values,
        val_df['price_p95_pred'].values
    )
    
    print(f"🎯 Mean IoU (validation): {val_iou.mean():.5f}")
    
    if 'is_narrow' in val_df.columns:
        mask = val_df['is_narrow'].values
        print(f"  ├─ Узкие товары: {val_iou[mask].mean():.5f} (n={mask.sum()})")
        print(f"  └─ Остальные:    {val_iou[~mask].mean():.5f}")

    test_p05_preds = np.column_stack([m.predict(test[features]) for m in models_p05])
    test_p95_preds = np.column_stack([m.predict(test[features]) for m in models_p95])
    
    test_p05_ens = np.quantile(test_p05_preds, 0.7, axis=1)
    test_p95_ens = np.quantile(test_p95_preds, 0.3, axis=1)
    test_p95_ens = np.maximum(test_p05_ens + eps, test_p95_ens)
    
    center_t = (test_p05_ens + test_p95_ens) / 2
    width_t  = test_p95_ens - test_p05_ens
    
    test['center'] = center_t
    test['width']  = width_t
    
    X_test_calib = test[calib_features].fillna(0).astype(float)
    c_test = c_calibrator.predict(X_test_calib)
    c_test = np.clip(c_test, 0.4, 1.2)
    
    test['price_p05'] = test['center'] - test['width'] * c_test / 2
    test['price_p95'] = test['center'] + test['width'] * c_test / 2
    test['price_p95'] = np.maximum(test['price_p05'] + eps, test['price_p95'])

    submission = test[['row_id', 'price_p05', 'price_p95']]
    submission

    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
