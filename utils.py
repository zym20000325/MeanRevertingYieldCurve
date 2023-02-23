import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


YumengZhangAPIkey = "qUL_zooxYcHueGAiB-D-"
quandl.ApiConfig.api_key = YumengZhangAPIkey


def get_yield_curve(start_date = '2001-08-01', end_date = '2023-02-01'):

    df_yc = quandl.get("USTREASURY/YIELD")
    
    df_yc_m = df_yc.resample('M').last()
    df_yc_m = df_yc_m[start_date:end_date]
    
    df_yc_m['2 MO_'] = (df_yc_m['1 MO'] + df_yc_m['3 MO'])/2
    df_yc_m['2 MO'] = df_yc_m['2 MO'].fillna(df_yc_m['2 MO_'])
    df_yc_m = df_yc_m.drop(['30 YR','2 MO_'], axis=1)

    df_yc_m .columns = ['1m', '2m', '3m', '6m', '1y', '2y', '3y','5y', '7y', '10y', '20y']

    return df_yc_m


def plot_yield_curve(df_):

    df = df_.copy()
    df.columns = ['1 month', '2 month', '3 month', '6 month', '1 year', '2 year', '3 year','5 year', '7 year', '10 year', '20 year']
    df.plot(figsize=(20, 8))

    plt.title('US Treasury Yield Curve Rates',fontsize=16)
    plt.ylabel('Yield Curve Rate')
    plt.xlabel('Date')
    plt.legend(loc='upper right')
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)
    plt.show();


def spot_rate(zcb, tenor):

    times = np.arange(tenor, 0, step=-0.5)[::-1]

    if times.shape[0]==0:
        sr = None

    else:
        r = np.interp(times, zcb.index.values, zcb.values) # Linear interpolation
        coupons_pv_x = 0.5*np.exp(-r*times).sum()
        final_pv = np.exp(-tenor*r[-1])
        sr = (1.0 - final_pv) / coupons_pv_x # Solves x + c * delta = 1.0
        
    return sr


def compute_spot_rates(zcb_rates):

    spot = zcb_rates.copy()

    for curve in zcb_rates.columns:

        zcb = zcb_rates[curve]

        for tenor, rate in zcb.items():
            if tenor>0.001:
                spot[curve][tenor] = spot_rate(zcb, tenor)

    return spot


def bond_price(zcb, coupon_rate, tenor):
    
    times = np.arange(tenor, 0, step=-0.5)[::-1]

    if times.shape[0]==0:
        p = 1.0
    else:
        r = np.interp(times, zcb.index.values, zcb.values) # Linear interpolation
        p = np.exp(-tenor*r[-1]) + 0.5 * coupon_rate * np.exp(-r*times).sum()

    return p


def compute_zcb_curve(spot_rates_curve):

    zcb_rates = spot_rates_curve.copy()

    for curve in spot_rates_curve.columns:
        spot = spot_rates_curve[curve]

        for tenor, spot_rate in spot.items():

            if tenor > 0.001:

                times = np.arange(tenor-0.5, 0, step=-0.5)[::-1]
                coupon_half_yr = 0.5*spot_rate

                z = np.interp(times, zcb_rates[curve].index.values, zcb_rates[curve].values) # Linear interpolation
                preceding_coupons_val = (coupon_half_yr*np.exp(-z*times)).sum()

                zcb_rates[curve][tenor] = -np.log((1-preceding_coupons_val)/(1+coupon_half_yr))/tenor

    return zcb_rates


def compute_zero_coupon(df):

    tenor = [1/12, 1/6, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20]
    cols = df.columns.to_list()

    df_zero = df.T
    df_zero = df_zero/100
    df_zero.index = tenor
    df_zero.index.names = ['Tenor']

    for c in df_zero.columns.to_list():

        df_zero[c] = compute_spot_rates(df_zero[[c]])
        df_zero[c] = compute_zcb_curve(df_zero[[c]])

    df_zero = df_zero*100

    df_zero = df_zero.T
    df_zero.columns = cols

    df_zero.columns = ['1m_0', '2m_0', '3m_0', '6m_0', '1y_0', '2y_0', '3y_0','5y_0', '7y_0', '10y_0', '20y_0']

    return df_zero


def compute_forward_yield(df):

    x_list = ['0', '1', '2', '5', '11', '23', '35', '59', '83', '119', '239']

    df_forward = df.copy()
    df_forward.columns = x_list

    for c in x_list[1:]:
        
        df_forward[c] = ((int(c)+1)*df_forward[c] - df_forward['0'])/int(c)

    df_forward.columns = ['1m_f', '2m_f', '3m_f', '6m_f', '1y_f', '2y_f', '3y_f','5y_f', '7y_f', '10y_f', '20y_f']

    return df_forward


def plot_yield_comparison(df_yc_m,df_zero,df_forward):

    plt.figure(figsize=(20, 8))

    plt.plot(df_yc_m['20y'][:'2012'], label = "Yield Curve (3 Month)")
    plt.plot(df_zero['20y_0'][:'2012'], label = "Zero Coupon Yield Curve (3 Month)")
    plt.plot(df_forward['20y_f'][:'2012'], label = "Forward Yield Curve (3 Month)")
    
    plt.title('Comparison of Yield Curve, Zero Coupon Yield Curve, and Forward Yield Curve', fontsize=16)
    plt.ylabel('Yield Curve Rate')
    plt.xlabel('date')
    plt.legend()
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)

    plt.show();


def transpose_to_tenor(df):

    df1 = df.T
    df1.index = [1, 2, 3, 6, 12, 24, 36, 60, 84, 120, 240]
    df1.index.names = ['Tenor']

    return df1


def plot_zero_coupon_comparison(df_zero):

    df_zero1 = transpose_to_tenor(df_zero)

    plt.figure(figsize=(20, 8))

    plt.plot(df_zero1['2001-12-31'], label = "December 2001")
    plt.plot(df_zero1['2003-12-31'], label = "December 2003")
    plt.plot(df_zero1['2009-12-31'], label = "December 2009")
    plt.plot(df_zero1['2012-12-31'], label = "December 2012")
    plt.plot(df_zero1['2015-12-31'], label = "December 2015")
    plt.plot(df_zero1['2020-12-31'], label = "December 2020")
    plt.plot(df_zero1['2021-12-31'], label = "December 2021")

    plt.title('Zero Coupon Yield Curve of Treasury with Different Maturities',fontsize=16)
    plt.ylabel('Zero Coupon Yield Curve Rate')
    plt.xlabel('Maturity (months)')
    plt.legend() 
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)

    plt.show();


def plot_forward_yield_comparison(df_yc_m,df_forward):

    plt.figure(figsize=(20, 8),constrained_layout=True)

    plt.subplot(2, 2, 1)

    plt.plot(df_yc_m['3y']['2012-01':'2012-12'], label = "Yield Curve (2012)")
    plt.plot(df_forward['3y_f']['2012-01':'2012-12'], label = "Forward Yield Curve (2012)")

    plt.title('3 Year Maturity')
    plt.ylabel('Yield Curve Rate')
    plt.xlabel('Date')
    plt.legend() 
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)

    plt.subplot(2, 2, 2)

    plt.plot(df_yc_m['5y']['2011-01':'2011-12'], label = "Yield Curve (2011)")
    plt.plot(df_forward['5y_f']['2011-01':'2011-12'], label = "Forward Yield Curve (2011)")

    plt.title('5 Year Maturity')
    plt.ylabel('Yield Curve Rate')
    plt.xlabel('Date')
    plt.legend() 
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)

    plt.subplot(2, 2, 3)

    plt.plot(df_yc_m['7y']['2010-01':'2010-12'], label = "Yield Curve (2010)")
    plt.plot(df_forward['7y_f']['2010-01':'2010-12'], label = "Forward Yield Curve (2010)")

    plt.title('7 Year Maturity')
    plt.ylabel('Yield Curve Rate')
    plt.xlabel('Date')
    plt.legend() 
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)

    plt.subplot(2, 2, 4)

    plt.plot(df_yc_m['10y']['2009-01':'2009-12'], label = "Yield Curve (2009)")
    plt.plot(df_forward['10y_f']['2009-01':'2009-12'], label = "Forward Yield Curve (2009)")

    plt.title('10 Year Maturity')
    plt.ylabel('Yield Curve Rate')
    plt.xlabel('Date')
    plt.legend() 
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)

    plt.suptitle('Comparison between Yield Curve and Forward Yield Curve', fontsize=16)

    plt.show()


def unconditional_average(df, start_date = '2002-08-31'):

    df_average = df.expanding().mean()
    
    return df_average[start_date:]


def unconditional_boxcar(df, window, start_date = '2002-08-31'):

    df_boxcar = df.rolling(window).mean()
    df_boxcar = df_boxcar[start_date:]

    return df_boxcar


def exponential_decay(df_, decay_coefficient):

    df = df_.copy()

    for i in range(len(df)):
        if i > 0:
            
            delta_t = (df.index.to_list()[i] - df.index.to_list()[i-1]).days/30
            w = np.exp((-1) * decay_coefficient * delta_t)

            df.iloc[i] = w * df.iloc[i-1] + (1-w) * df.iloc[i]

    return df


def unconditional_exponential_decay(df_, decay_coefficient, window = -1, start_date = '2002-08-31'):

    df = df_.copy()

    for i in range(len(df)):

        if window != -1:
            if i >= window:

                df_window = df_.iloc[(i-window):(i+1)].copy()
                df_window = exponential_decay(df_window, decay_coefficient)

                df.iloc[i] = df_window.iloc[-1]
        
        else:
            if i > 0:

                df_window = df_.iloc[:(i+1)].copy()
                df_window = exponential_decay(df_window, decay_coefficient)

                df.iloc[i] = df_window.iloc[-1]
    
    return df[start_date:]


def plot_unconditional_yc_comparision(df_forward, df_uncon_average, df_uncon_boxcar, df_uncon_exp_decay, df_uncon_exp_decay2):

    plt.figure(figsize=(20, 8))

    plt.plot(df_forward['1m_f'], label = "Original Forward Yield Curve")
    plt.plot(df_uncon_average['1m_f'], label = "Average")
    plt.plot(df_uncon_boxcar['1m_f'], label = "Boxcar")
    plt.plot(df_uncon_exp_decay['1m_f'], label = "Exponential Decay")
    plt.plot(df_uncon_exp_decay2['1m_f'], label = "Exponential Decay with Window")

    plt.title('Unconditional Yield Curve Comparison (1 Month Maturity)', fontsize=16)
    plt.ylabel('Yield Curve Rate')
    plt.xlabel('Date')
    plt.legend()
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)

    plt.show();


def plot_unconditional_yc_comparision2(df_forward, df_uncon_average, df_uncon_boxcar, df_uncon_exp_decay, df_uncon_exp_decay2):

    plt.figure(figsize=(20, 8))

    plt.plot(df_forward.T['2015-07-31'], label = "Original Forward Yield Curve")
    plt.plot(df_uncon_average.T['2015-07-31'], label = "Average")
    plt.plot(df_uncon_boxcar.T['2015-07-31'], label = "Boxcar")
    plt.plot(df_uncon_exp_decay.T['2015-07-31'], label = "Exponential Decay")
    plt.plot(df_uncon_exp_decay2.T['2015-07-31'], label = "Exponential Decay with Window")

    plt.title('Unconditional Yield Curve Comparison (July 2015)', fontsize=16)
    plt.ylabel('Yield Curve Rate')
    plt.xlabel('Maturity (months)')
    plt.legend()
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)

    plt.show();