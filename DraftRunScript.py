import pandas as pd
import scsavailability as scs
    
from scsavailability import features as feat, model as md, plotting as pt, results as rs

path = 'C:/Users/Jamie.williams/OneDrive - Newton Europe Ltd/Castle Donnington/Data/'

at = pd.read_csv(path+'active_totes_20201210.csv')
av = pd.read_csv(path+'Availability_with_Grey&Blue_1811-0912.csv',names = ["timestamp","Pick Station","Availability","Blue Tote Loss","Grey Tote Loss"])
scs_raw = pd.read_csv(path + 'Faults20_11-10_12.csv')

at = feat.pre_process_AT(at)
av = feat.pre_process_av(av)
fa,unmapped = feat.preprocess_faults(scs_raw,remove_same_location_faults = True)

fa_floor = feat.floor_shift_time_fa(fa, shift=0)

df,fa_PTT = feat.create_PTT_df(fa_floor,at,av)
df = feat.log_totes(df) 

X,y = md.gen_feat_var(df,target = 'Availability', features = ['Totes','Faults'])
X_train, X_test, y_train, y_test = md.split(X,y,split_options = {'test_size': 0.3,
                                                                 'random_state': None})
R2_cv,R2_OOS,Coeff = md.run_OLS(X_train = X_train,y_train = y_train,X_test = X_test,y_test=y_test, n = 5)

Output = rs.create_output(fa_PTT,Coeff)

