import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
pd.options.mode.chained_assignment = None 


YumengZhangAPIkey = "qUL_zooxYcHueGAiB-D-"
quandl.ApiConfig.api_key = YumengZhangAPIkey


def get_yield_curve(start_date = '2001-08-01', end_date = '2022-08-01'):

    df_yc = quandl.get("USTREASURY/YIELD")
    
    df_yc_m = df_yc.resample('M').last()
    df_yc_m = df_yc_m[start_date:end_date]
    
    df_yc_m['2 MO_'] = (df_yc_m['1 MO'] + df_yc_m['3 MO'])/2
    df_yc_m['2 MO'] = df_yc_m['2 MO'].fillna(df_yc_m['2 MO_'])
    df_yc_m = df_yc_m.drop(['30 YR','2 MO_'], axis=1)

    df_yc_m.columns = ['1m', '2m', '3m', '6m', '1y', '2y', '3y','5y', '7y', '10y', '20y']

    return df_yc_m


def get_fed_fund_rate(start_date = '2001-08-01', end_date = '2022-08-01'):

    fed_funds_rate = quandl.get("FRED/DFF")
    fed_funds_rate = fed_funds_rate.resample('M').last()
    fed_funds_rate = fed_funds_rate[start_date:end_date]
    fed_funds_rate.columns = ['ffr']

    fed_funds_rate['ffr'] = fed_funds_rate['ffr']/100 
    fed_funds_rate['ffr'] = (1+fed_funds_rate['ffr'])**(1/12)-1
    fed_funds_rate['ffr'] = fed_funds_rate['ffr'] + 0.005           # transaction cost

    return fed_funds_rate


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


def transpose_to_tenor_year(df):

    df1 = df.T
    df1.index = [1/12, 1/6, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20]
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


def plot_unconditional_yc_comparision(df_origin, df_uncon_average, df_uncon_boxcar, df_uncon_exp_decay, df_uncon_exp_decay2,rate):

    plt.figure(figsize=(20, 8))

    if rate == 'f':

        plt.plot(df_origin['1m_f'], label = "Original Forward Yield Curve")
        plt.plot(df_uncon_average['1m_f'], label = "Average")
        plt.plot(df_uncon_boxcar['1m_f'], label = "Boxcar")
        plt.plot(df_uncon_exp_decay['1m_f'], label = "Exponential Decay")
        plt.plot(df_uncon_exp_decay2['1m_f'], label = "Exponential Decay with Window")
    
    elif rate == 'z':

        plt.plot(df_origin['1m_0'], label = "Original Zero Coupon Rate Curve")
        plt.plot(df_uncon_average['1m_0'], label = "Average")
        plt.plot(df_uncon_boxcar['1m_0'], label = "Boxcar")
        plt.plot(df_uncon_exp_decay['1m_0'], label = "Exponential Decay")
        plt.plot(df_uncon_exp_decay2['1m_0'], label = "Exponential Decay with Window")

    plt.title('Unconditional Yield Curve Comparison (1 Month Maturity)', fontsize=16)
    plt.ylabel('Yield Curve Rate')
    plt.xlabel('Date')
    plt.legend()
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)

    plt.show();


def plot_unconditional_yc_comparision2(df_origin, df_uncon_average, df_uncon_boxcar, df_uncon_exp_decay, df_uncon_exp_decay2,rate):

    plt.figure(figsize=(20, 8))

    if rate == 'f':
        plt.plot(df_origin.T['2015-07-31'], label = "Original Forward Yield Curve")
    
    elif rate == 'z':
        plt.plot(df_origin.T['2015-07-31'], label = "Original Zero Coupon Rate")

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


def fitted_curvature(x,y):

    # Constructing piecewise cubic spline function
    f = interp1d(x, y, kind='cubic')

    xnew = np.linspace(x[0], x[-1], 200)

    # Compute the curvature using the formula
    dfdx = np.gradient(f(xnew), xnew)
    d2fdx2 = np.gradient(dfdx, xnew)
    
    k = np.abs(d2fdx2) / (1 + dfdx**2)**(3/2)

    # plt.plot(x,y,'o', label = 'Original Data Points')
    # plt.plot(xnew, f(xnew), label = 'Fitted Curve')
    plt.plot(xnew, k, label = 'Curvature')
    plt.xlabel('x')
    plt.ylabel('Curvature')
    plt.legend()
    plt.show()

    # return k


def get_curvature_diff(df_curvature, quantile = 0.25, plot = False):

    df_curvature_diff = pd.DataFrame(index = df_curvature.index)
    df_curvature_diff['diff'] = df_curvature['origin_curvature'] - df_curvature['uncon_curvature']
    df_curvature_diff['diff'] = df_curvature_diff['diff'].abs()

    if plot == True:

        plt.figure(figsize=(20, 8))
        plt.plot(df_curvature_diff['diff'], label = "Curvature Diff")

        plt.title('Absolute Value of Difference Between Original Curvature and Unconditional Curvature', fontsize=16)
        plt.ylabel('Absolute Value of Curvature Difference')
        plt.xlabel('Date')
        plt.legend()
        plt.gca().set_facecolor('lightgray')
        plt.grid(True)

        plt.show();

    return df_curvature_diff['diff'].quantile(quantile)


def three_bond_curvature(df_origin, df_uncon, three_bonds, bound, rate):

    month_dict1_f = {'1m':'1m_f', '2m':'2m_f', '3m':'3m_f', '6m':'6m_f', '1y':'1y_f', '2y':'2y_f','3y':'3y_f', '5y':'5y_f', '7y':'7y_f','10y':'10y_f','20y':'20y_f'}
    month_dict1_0 = {'1m':'1m_0', '2m':'2m_0', '3m':'3m_0', '6m':'6m_0', '1y':'1y_0', '2y':'2y_0','3y':'3y_0', '5y':'5y_0', '7y':'7y_0','10y':'10y_0','20y':'20y_0'}

    # month_dict2 = {'1m':1, '2m':2, '3m':3, '6m':6, '1y':12, '2y':24,'3y':36, '5y':60, '7y':84,'10y':120,'20y':240}
    month_dict2 = {'1m':1/12, '2m':2/12, '3m':3/12, '6m':6/12, '1y':1, '2y':2,'3y':3, '5y':5, '7y':7,'10y':10,'20y':20}

    X = three_bonds[0]
    Y = three_bonds[1]
    Z = three_bonds[2]

    if rate == 'f':

        r_x = month_dict1_f[X]
        r_y = month_dict1_f[Y]
        r_z = month_dict1_f[Z]
    
    elif rate == 'z':

        r_x = month_dict1_0[X]
        r_y = month_dict1_0[Y]
        r_z = month_dict1_0[Z]

    x = month_dict2[X]
    y = month_dict2[Y]
    z = month_dict2[Z]

    df_origin['origin_curvature'] = ((df_origin[r_y] - df_origin[r_x])/(y-x)) - ((df_origin[r_z] - df_origin[r_y])/(z-y))
    df_uncon['uncon_curvature'] = ((df_uncon[r_y] - df_uncon[r_x])/(y-x)) - ((df_uncon[r_z] - df_uncon[r_y])/(z-y))

    df_origin = df_origin.join(df_uncon[['uncon_curvature']])

    df_curvature = df_origin[[r_x,r_y,r_z,'origin_curvature','uncon_curvature']]

    if bound != 0.0:
        quantile_value = get_curvature_diff(df_curvature, quantile = bound, plot = False)

        df_curvature['upper'] = df_curvature['uncon_curvature'] + quantile_value
        df_curvature['lower'] = df_curvature['uncon_curvature'] - quantile_value

    df_origin = df_origin.drop(columns=['origin_curvature','uncon_curvature'])
    df_uncon = df_uncon.drop(columns=['uncon_curvature'])

    return df_curvature.dropna()


def plot_curvature(df_curvature, title, rate, bound = False):

    plt.figure(figsize=(20, 8))

    if rate == 'f':
        plt.plot(df_curvature['origin_curvature'], label = "Forward Yield Curvature")
    elif rate == 'z':
        plt.plot(df_curvature['origin_curvature'], label = "Zero Coupon Rate Curvature")

    plt.plot(df_curvature['uncon_curvature'], label = "Unconditional Yield Curvature")

    if rate == 'f':
        plt.title('Comparison of Curvature - Forward Yield Curve VS Unconditional Yield Curve ('+title+')',fontsize=16)
    elif rate == 'z':
        plt.title('Comparison of Curvature - Zero Coupon Rate VS Unconditional Yield Curve ('+title+')',fontsize=16)

    if bound == True:
        plt.plot(df_curvature['upper'], label = "Upper Bound", color = 'grey')
        plt.plot(df_curvature['lower'], label = "Lower Bound", color = 'grey')

    plt.ylabel('Curvature')
    plt.xlabel('Date')
    plt.legend() 
    plt.gca().set_facecolor('lightgray')
    plt.grid(True)

    plt.show();


def bond_price(zcb, coupon_rate, tenor):
    
    times = np.arange(tenor, 0, step=-0.5)[::-1]

    if times.shape[0]==0:
        p = 1.0
    else:
        r = np.interp(times, zcb.index.values, zcb.values) # Linear interpolation
        # p = np.exp(-tenor*r[-1]) + 0.5 * coupon_rate * np.exp(-r*times).sum()
        p = np.exp(-tenor*r[-1]) + 0.5 * coupon_rate/100 * np.exp(-r*times).sum()

    return p


def calculate_bond_price(df_zero, df_yc_m, tenor):

    df_bond_price = pd.DataFrame(index = df_zero.index)
    date_list = df_bond_price.index.tolist()

    df_bond_price['old_bond_price'] = 0.0
    df_bond_price['new_bond_price'] = 0.0

    df_zero_t = transpose_to_tenor_year(df_zero)/100
    # df_zero_t = transpose_to_tenor(df_zero)/100

    for i in range(len(df_bond_price)):

        zcb = df_zero_t[date_list[i]]

        # if(i==0):
        #     print(zcb)

        # df_bond_price['old_bond_price'][i] = bond_price(zcb, df_yc_m['5y'][i]/100, tenor=tenor)
        # df_bond_price['new_bond_price'][i] = bond_price(zcb, df_yc_m['5y'][i]/100, tenor=tenor-1/12)

        df_bond_price['old_bond_price'][i] = bond_price(zcb, df_yc_m['5y'][i], tenor=tenor)
        df_bond_price['new_bond_price'][i] = bond_price(zcb, df_yc_m['5y'][i], tenor=tenor-1/12)

    df_bond_price['old_bond_price'] = df_bond_price['old_bond_price'].shift(1)
    df_bond_price = df_bond_price.dropna()

    return df_bond_price


def trade(df_origin, df_uncon, df_yc_m, df_fed_fund_rate,three_bonds, bound = 0.0):

    df_curvature = three_bond_curvature(df_origin, df_uncon, three_bonds, bound = bound,rate='z')
    df_curvature = df_curvature.join(df_fed_fund_rate)

    # month_dict1 = {'1m':'1m_f', '2m':'2m_f', '3m':'3m_f', '6m':'6m_f', '1y':'1y_f', '2y':'2y_f','3y':'3y_f', '5y':'5y_f', '7y':'7y_f','10y':'10y_f','20y':'20y_f'}
    month_dict1 = {'1m':'1m_0', '2m':'2m_0', '3m':'3m_0', '6m':'6m_0', '1y':'1y_0', '2y':'2y_0','3y':'3y_0', '5y':'5y_0', '7y':'7y_0','10y':'10y_0','20y':'20y_0'}
    # month_dict2 = {'1m':1/12, '2m':2/12, '3m':3/12, '6m':6/12, '1y':1, '2y':2,'3y':3, '5y':5, '7y':7,'10y':10,'20y':20}
    month_dict2 = {'1m':1, '2m':2, '3m':3, '6m':6, '1y':12, '2y':24,'3y':36, '5y':60, '7y':84,'10y':120,'20y':240}

    X = three_bonds[0]
    Y = three_bonds[1]
    Z = three_bonds[2]

    r_x = month_dict1[X]
    r_y = month_dict1[Y]
    r_z = month_dict1[Z]

    x = month_dict2[X]
    y = month_dict2[Y]
    z = month_dict2[Z]

    K = 800

    df_origin_ = df_origin.drop(columns=['origin_curvature'])

    x_price = calculate_bond_price(df_origin_, df_yc_m, x/12) 
    y_price = calculate_bond_price(df_origin_, df_yc_m, y/12)
    z_price = calculate_bond_price(df_origin_, df_yc_m, z/12)

    df_curvature['pnl'] = 0.0
    df_curvature['borrowing_cost'] = 0.0

    index_list = df_curvature.index.values.tolist()

    for i in range(len(df_curvature)):

        # p_x = (1000/(x-1))*((1+df_curvature[r_x][i]/100)**((x-1)/12))
        # p_y = (1000/(y-1))*((1+df_curvature[r_y][i]/100)**((y-1)/12))
        # p_z = (1000/(z-1))*((1+df_curvature[r_z][i]/100)**((z-1)/12))

        # p_x = (1000/(x-1))*((1+df_curvature[r_x][i]/100)**(1/12))
        # p_y = (1000/(y-1))*((1+df_curvature[r_y][i]/100)**(1/12))
        # p_z = (1000/(z-1))*((1+df_curvature[r_z][i]/100)**(1/12))

        date = df_curvature.index.values[i]
        # print(date)

        # p_x = (1000/(x-1))*(x_price.loc[date]['new_bond_price']/x_price.loc[date]['old_bond_price'] - 1)
        # p_y = (1000/(y-1))*(y_price.loc[date]['new_bond_price']/y_price.loc[date]['old_bond_price'] - 1)
        # p_z = (1000/(z-1))*(z_price.loc[date]['new_bond_price']/z_price.loc[date]['old_bond_price'] - 1) 

        p_x = (1000)*(x_price.loc[date]['new_bond_price']/x_price.loc[date]['old_bond_price']-1)
        p_y = (1000)*(y_price.loc[date]['new_bond_price']/y_price.loc[date]['old_bond_price']-1)
        p_z = (1000)*(z_price.loc[date]['new_bond_price']/z_price.loc[date]['old_bond_price']-1) 

        if bound == 0.0:

            if df_curvature['origin_curvature'][i] > df_curvature['uncon_curvature'][i]:

                # df_curvature['pnl'][i] = p_y - p_x - p_z   # 反过来  x+z-y 存的钱 乘 rate  +
                # df_curvature['borrowing_cost'][i] = 800 * df_curvature['ffr'][i]

                df_curvature['pnl'][i] = p_x + p_z - p_y
                # df_curvature['borrowing_cost'][i] = (1000/(y-1)-1000/(x-1)-1000/(z-1)) * df_curvature['ffr'][i]
                df_curvature['borrowing_cost'][i] = 800 * df_curvature['ffr'][i]

                df_curvature['pnl'][i] = df_curvature['pnl'][i] + df_curvature['borrowing_cost'][i]

            elif df_curvature['origin_curvature'][i] < df_curvature['uncon_curvature'][i]:

                # df_curvature['pnl'][i] = p_x + p_z - p_y   # 1000/(y-1) - x - z 借的钱   +
                # df_curvature['borrowing_cost'][i] = -200 * df_curvature['ffr'][i]

                df_curvature['pnl'][i] = p_y - p_x - p_z
                # df_curvature['borrowing_cost'][i] = (1000/(x-1)+1000/(z-1)-1000/(y-1)) * df_curvature['ffr'][i]
                df_curvature['borrowing_cost'][i] = -200 * df_curvature['ffr'][i]

                df_curvature['pnl'][i] = df_curvature['pnl'][i] + df_curvature['borrowing_cost'][i]
            
        else:

            if df_curvature['origin_curvature'][i] > df_curvature['upper'][i]:

                # df_curvature['pnl'][i] = p_y - p_x - p_z   # 反过来  x+z-y 存的钱 乘 rate  +
                # df_curvature['borrowing_cost'][i] = 800 * df_curvature['ffr'][i]

                df_curvature['pnl'][i] = p_x + p_z - p_y
                # df_curvature['borrowing_cost'][i] = (1000/(y-1)-1000/(x-1)-1000/(z-1)) * df_curvature['ffr'][i]
                df_curvature['borrowing_cost'][i] = 800 * df_curvature['ffr'][i]

                df_curvature['pnl'][i] = df_curvature['pnl'][i] + df_curvature['borrowing_cost'][i]

            elif df_curvature['origin_curvature'][i] < df_curvature['lower'][i]:

                # df_curvature['pnl'][i] = p_x + p_z - p_y   # 1000/(y-1) - x - z 借的钱   +
                # df_curvature['borrowing_cost'][i] = -200 * df_curvature['ffr'][i]

                df_curvature['pnl'][i] = p_y - p_x - p_z
                # df_curvature['borrowing_cost'][i] = (1000/(x-1)+1000/(z-1)-1000/(y-1)) * df_curvature['ffr'][i]
                df_curvature['borrowing_cost'][i] = -200 * df_curvature['ffr'][i]

                df_curvature['pnl'][i] = df_curvature['pnl'][i] + df_curvature['borrowing_cost'][i]
            

    df_curvature['cumulative_pnl'] = np.cumsum(df_curvature['pnl'])
    df_curvature['return'] = df_curvature['pnl']/K 
    df_curvature['cum_return'] = np.cumprod(1+df_curvature['return']) - 1

    return df_curvature


def performance_metrics(df, name, returns_plot_show = False, pnl_plot_show = False):
    
    # returns = df['cumulative_return'].to_frame()
    returns = df['return'].to_frame()
    cumulative_returns = df['cum_return'].to_frame()
    cumulative_pnl = df['cumulative_pnl'].to_frame()
    
    metrics = pd.DataFrame(index=returns.columns)
    
    metrics['Mean'] = round(returns.mean(),6)*12
    metrics['Vol'] = round(returns.std(),6)*np.sqrt(12)
    metrics['Sharpe'] = round(returns.mean()/returns.std(),4)*np.sqrt(12)

    metrics['Min'] = min(df['return'])
    metrics['Max'] = max(df['return'])
        
    metrics['Max Drawdown'] = (cumulative_returns['cum_return'] - 
                               cumulative_returns['cum_return'].rolling(len(cumulative_returns['cum_return']), 
                                                                 min_periods=1).max()).min()
        
    if returns_plot_show == True:
        cumulative_returns.plot(title="Cumulative Net Return", ylabel = "cumulative net return (%)" ,figsize = (12,5))
        
    if pnl_plot_show == True:
        cumulative_pnl.plot(title="Cumulative PnL",figsize = (12,5))

    metrics = metrics.rename(index={'return': name})
        
    return metrics