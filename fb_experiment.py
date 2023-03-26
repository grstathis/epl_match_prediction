import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from os import listdir
from os.path import isfile, join
from xgboost import XGBClassifier
import pickle


class FootballETL:

    mypath = 'premier_league_csv/'
    bookmaker_l = ['B365','BW','IW','VC','PS','WH']
    bookmaker_col_l = ['B365H', 'B365A', 'B365D', 'BWH', 'BWA', 'BWD', 'IWH', 'IWA', 'IWD',
                       'PSH', 'PSA', 'PSD', 'WHH', 'WHA', 'WHD', 'VCH', 'VCA', 'VCD']
    feat_l = ['B365H', 'B365A', 'B365D', 'BWH_diff', 'IWH_diff', 'PSH_diff', 'WHH_diff', 'VCH_diff',
              'BWA_diff', 'IWA_diff', 'PSA_diff', 'WHA_diff', 'VCA_diff',
              'BWD_diff', 'IWD_diff', 'PSD_diff', 'WHD_diff', 'VCD_diff']

    def read_csv_files(self, mypath):
        ret_df = pd.DataFrame()
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for f in onlyfiles:
            df = pd.read_csv(mypath + f)
            if len(df.iloc[0]['Date']) == 10:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            else:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
            df = df.sort_values('Date')
            print('fyear:', int(df.iloc[0]['Date'].year))
            df['FYear'] = int(df.iloc[0]['Date'].year)
            ret_df = ret_df.append(df, ignore_index=True)
        ret_df = ret_df.sort_values('Date')
        return ret_df

    def read_csv_files_mvalue(self, mypath):
        ret_df = pd.DataFrame()
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for f in onlyfiles:
            df = pd.read_csv(mypath + f)
            df['FYear'] = int(f[10:-4])
            ret_df = ret_df.append(df, ignore_index=True)
        return ret_df
    
    def add_date_types(self, df):
        df['Year'] = pd.DatetimeIndex(df['Date']).year
        df['Month'] = pd.DatetimeIndex(df['Date']).month
        return df

    def create_diff_col(self, df):
        df['BWH_diff'] = df['B365H'] - df['BWH']
        df['IWH_diff'] = df['B365H'] - df['IWH']
        df['PSH_diff'] = df['B365H'] - df['PSH']
        df['WHH_diff'] = df['B365H'] - df['WHH']
        df['VCH_diff'] = df['B365H'] - df['VCH']

        df['BWA_diff'] = df['B365A'] - df['BWA']
        df['IWA_diff'] = df['B365A'] - df['IWA']
        df['PSA_diff'] = df['B365A'] - df['PSA']
        df['WHA_diff'] = df['B365A'] - df['WHA']
        df['VCA_diff'] = df['B365A'] - df['VCA']

        df['BWD_diff'] = df['B365D'] - df['BWD']
        df['IWD_diff'] = df['B365D'] - df['IWD']
        df['PSD_diff'] = df['B365D'] - df['PSD']
        df['WHD_diff'] = df['B365D'] - df['WHD']
        df['VCD_diff'] = df['B365D'] - df['VCD']
        return df

    def return_feat_list(self, col_ref):
        feat_l = []

        for side in ['H', 'A', 'D']:
            for bookm in self.bookmaker_l:
                if col_ref != bookm:
                    feat_l.append(col_ref+side+'_' + bookm + side + '_diff')
        for side in ['H', 'A', 'D']:
            feat_l.append(col_ref + side)

        return feat_l

    def create_diff_col_all(self, df):
        i = 0
        for bookl in self.bookmaker_l:
            i = i + 1
            for bookm in self.bookmaker_l[i:]:
                for side in ['H', 'A', 'D']:
                    df[bookl+side+'_' + bookm + side + '_diff'] = df[bookl+side] - df[bookm+side]

        return df    
    
    def create_diff_col_ref(self, df, col_ref):

        for side in ['H', 'A', 'D']:
            for bookm in self.bookmaker_l:
                if col_ref != bookm:
                    df[col_ref+side+'_' + bookm + side + '_diff'] = df[col_ref+side] - df[bookm+side]

        # for col in df.columns:
        #     if len(df[col].unique()) == 1:
        #         df.drop(col, inplace=True, axis=1)
        return df

    def add_ftr_binary(self, df):
        df.loc[:,'FTR_i'] = 0
        df.loc[df['FTR'] == 'H', 'FTR_i'] = 0
        df.loc[df['FTR'] == 'A', 'FTR_i'] = 1
        df.loc[df['FTR'] == 'D', 'FTR_i'] = 1
        df.loc[:,'FTR_a'] = 0
        df.loc[df['FTR'] == 'H', 'FTR_a'] = 0
        df.loc[df['FTR'] == 'A', 'FTR_a'] = 1
        df.loc[df['FTR'] == 'D', 'FTR_a'] = -1
        return df

    def split_fyear(self, df, year, lookback, verbose=False):
        y_l = df['FYear'].unique()
        y_test = df[(df['FYear'] == year) & (df['Year'] == (year + 1))]['FTR_i']
        Y_test = df[(df['FYear'] == year) & (df['Year'] == (year + 1))][['FTR', 'FTR_i']]
        X_test = df[(df['FYear'] == year) & (df['Year'] == (year + 1))][self.feat_l]
        X_test_odds = df[(df['FYear'] == year) & (df['Year'] == (year + 1))][['B365H', 'B365A', 'B365D',
                                                                              'HomeTeam', 'AwayTeam','Date']]
        if verbose:
            print(y_l[:(np.where(y_l == np.int64(year))[0][0])][lookback:], year)

        X_train = df[df['FYear'].isin(y_l[:(np.where(y_l == np.int64(year))[0][0])][lookback:])][self.feat_l].\
            append(df[(df['FYear'] == year) & (df['Year'] == year)][self.feat_l])
        y_train = df[df['FYear'].isin(y_l[:(np.where(y_l == np.int64(year))[0][0])][lookback:])]['FTR_i'].\
            append(df[(df['FYear'] == year) & (df['Year'] == year)]['FTR_i'])
        return X_train, y_train, X_test, y_test, Y_test, X_test_odds

    def split_year(self, df, year, lookback, verbose=False):
        y_l = df['FYear'].unique()
        y_test = df[(df['FYear'] == year)]['FTR_i']
        Y_test = df[(df['FYear'] == year)][['FTR', 'FTR_i']]
        X_test = df[(df['FYear'] == year)][self.feat_l]
        X_test_odds = df[(df['FYear'] == year)][['B365H', 'B365A', 'B365D', 'HomeTeam', 'AwayTeam','Date']]
        if verbose:
            print(y_l[:(np.where(y_l == year)[0][0])][lookback:], year)

        X_train = df[df['FYear'].isin(y_l[:(np.where(y_l == year)[0][0])][lookback:])][self.feat_l]
        y_train = df[df['FYear'].isin(y_l[:(np.where(y_l == year)[0][0])][lookback:])]['FTR_i']
        return X_train, y_train, X_test, y_test, Y_test, X_test_odds
    
    
    def model_fit(self, model, X_train, y_train, scale, pca_apply, quantile_tr):
        scaler = None
        pca = None
        qt = None

        if quantile_tr:
            qt = preprocessing.QuantileTransformer(n_quantiles=50, random_state=0, output_distribution='normal')
            X_train[['B365H', 'B365D', 'B365A']] = qt.fit_transform(X_train[['B365H', 'B365D', 'B365A']])

        if scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)

        if pca_apply:
            pca = PCA(n_components=12)
            X_train = pca.fit_transform(X_train)

        model.fit(X_train, y_train)
        return model, scaler, pca, qt

    def model_predict(self, model, X_test, scale, pca_apply, quant_tr, scaler=None, pca=None, qt=None, prob=False):

        if quant_tr:
            X_test[['B365H', 'B365D', 'B365A']] = qt.transform(X_test[['B365H', 'B365D', 'B365A']])

        if scale:
            X_test = scaler.transform(X_test)

        if pca_apply:
            X_test = pca.transform(X_test)

        predicted = model.predict(X_test)
        if prob:
            predicted = model.predict_proba(X_test)

        return predicted

    def model_evaluate(self, y_test, predicted, verbose=True, prob=False, threshold=0.0):
        predict_l = []
        if prob:
            for r in predicted:
                if r[0] > threshold:
                    predict_l.append(0)
                else:
                    predict_l.append(1)
            predicted = predict_l
        accuracy = accuracy_score(y_test, predicted, )
        precision = precision_score(y_test, predicted)
        recall = recall_score(y_test, predicted)
#         auc = roc_auc_score(y_test, predicted)
        if verbose:
            print(f'Mean accuracy score: {accuracy:.3}')
            print(f'Precision score: {precision:.3}')
            print(f'Recall score: {recall:.3}')
#             if prob:
#                 print(f'AUC score: {auc:.3}')
            print('confusion matrix:', confusion_matrix(y_test, predicted))
        return accuracy, precision, recall

    
    def play_the_game(self, predicted_l, predicted_prob_l, X_test, Y_test, odds_thres, bet, threshold, verbose, doubling=False):
        profit = 0
        games_played = 0
        hit = 0
        miss = 0
        init_bet = bet
        for i in range(X_test.shape[0]):
            
            pred = predicted_l[i]
            pred_prob = predicted_prob_l[i]
            if (pred_prob[1] > threshold):
                if (X_test.iloc[i]['B365A'] > odds_thres) & (X_test.iloc[i]['B365D'] > odds_thres):
                    odds_sum = X_test.iloc[i]['B365A'] + X_test.iloc[i]['B365D']
                    ratio_draw = X_test.iloc[i]['B365A'] / odds_sum
                    ratio_away = X_test.iloc[i]['B365D'] / odds_sum

                    if verbose:
                        print('odds away:', X_test.iloc[i]['B365A'], 'odds draw:', X_test.iloc[i]['B365D'])
                        print('ratio draw:', ratio_draw, 'ratio away:', ratio_away)
                        print(X_test.iloc[i][['Date','HomeTeam', 'AwayTeam']], 'result',Y_test.iloc[i]['FTR'])

                    games_played = games_played + 1
                    if Y_test.iloc[i]['FTR'] == 'A':
                        hit = hit + 1
                        profit = profit + ratio_away * bet * (X_test.iloc[i]['B365A']) - bet
                        bet = init_bet
                    elif Y_test.iloc[i]['FTR'] == 'D':
                        hit = hit + 1
                        profit = profit + ratio_draw * bet * (X_test.iloc[i]['B365D']) - bet
                        bet = init_bet
                    else:
                        miss = miss + 1
                        profit = profit - bet
                        if doubling:
                            bet = 2*bet
                    if verbose:
                        print('current profit:', profit)

            elif (pred_prob[0] >= threshold):
                
                if X_test.iloc[i]['B365H'] > odds_thres:
                    if verbose:
                        print('odds home:', X_test.iloc[i]['B365H'])
                        print(X_test.iloc[i][['Date','HomeTeam', 'AwayTeam']],'result',Y_test.iloc[i]['FTR'])

                    games_played = games_played + 1
                    if Y_test.iloc[i]['FTR'] == 'H':
                        hit = hit + 1
                        profit = profit + bet * (X_test.iloc[i]['B365H']) - bet
                        bet = init_bet
                    else:
                        miss = miss + 1
                        profit = profit - bet
                        if doubling:
                            bet = 2*bet
                    if verbose:
                        print('current profit:', profit)

        return profit, games_played, hit, miss

    def year_ml_walk(self, df, year, model, scale, pca_apply, quant_tr, lag, odds_threshold, prob_threshold, bet,
                     eval_print=False, show_games=False, savefile=None):
        X_train, y_train, X_test, y_test, Y_test, X_test_odds = self.split_fyear(df, year, lag)
        model, scaler, pca, qt = self.model_fit(model, X_train, y_train, scale=scale, pca_apply=pca_apply,
                                                quantile_tr=quant_tr)
        predicted_prob_l = self.model_predict(model, X_test, scale, pca_apply=pca_apply, quant_tr=quant_tr,
                                              scaler=scaler, pca=pca, qt=qt, prob=True)
        predicted_l = self.model_predict(model, X_test, scale, pca_apply=pca_apply, quant_tr=quant_tr,
                                         scaler=scaler, pca=pca, qt=qt, prob=False)
        self.model_evaluate(y_test, predicted_l, eval_print, False, 0.6)
        if savefile:
            pickle.dump(model, open(savefile+'.model', 'wb'))
            pickle.dump(scaler, open(savefile + '.scaler', 'wb'))
            pickle.dump(pca, open(savefile + '.pca', 'wb'))

        profit, games_played, hit, miss = self.play_the_game(predicted_l, predicted_prob_l, X_test_odds, Y_test,
                                                             odds_threshold, bet, prob_threshold, show_games)
        return profit, games_played, hit, miss

    def all_year_ml_walk(self, df, model, scale, pca_apply, quant_tr, lag, odds_threshold, prob_threshold, bet,
                         eval_print):
        res = []

        for year in [2020, 2019, 2018, 2017]:
            profit, games_played, hit, miss = self.year_ml_walk(df, year, model, scale, pca_apply, quant_tr, lag,
                                                                odds_threshold, prob_threshold, bet, eval_print, False)
            res.append([year, profit, games_played, hit, miss])

        return pd.DataFrame(res, columns=['year', 'profit', 'games_played', 'hit', 'miss'])

    
    def search_best_xb_params(self, fb_games_df, bet, eval_print, random_state, pca_apply,  write_to_file, verbose):
        max_prof = 0
        prob_res = 0
        odds_res = 0
        lag_res = 0
        max_depth_res = 0
        mchild_w_res = 0
        eta_res = 0
        scale = True
        quant_tr = False

        if write_to_file:
            with open(write_to_file, 'w') as f:
                f.write('')

        for prob_threshold in np.linspace(0.6, 0.9, 5):
            print('loop prob thres:', prob_threshold)
            for odds_threshold in np.linspace(2, 4, 5):
                for lag in [-1, -2, -3]:
                    for depth in [5, 6, 7]:
                        for min_child_weight in [1, 2, 3, 4]:
                            for eta in [0.35]:
                                xb = XGBClassifier(use_label_encoder=False, random_state=random_state,
                                                   max_depth=depth, min_child_weight=min_child_weight, eta=eta,
                                                   subsample=1,
                                                   colsample_bytree=1, eval_metric='logloss')
                                model = xb
                                res_df = self.all_year_ml_walk(fb_games_df, model, scale, pca_apply, quant_tr, lag,
                                                               odds_threshold, prob_threshold, bet, eval_print)
                                if len(res_df[res_df['profit'] < 0]) > 0:
                                    profit = 0
                                else:
                                    profit = res_df['profit'].sum()
                                    if verbose:
                                        print(profit, odds_threshold, prob_threshold, lag, depth, min_child_weight, eta)
                                        print(res_df)
                                    if write_to_file:
                                        with open(write_to_file, 'a') as f:
                                            f.write("profit = " + str(profit) + "\n")
                                            f.write("odds_threshold = " + str(odds_threshold) + "\n")
                                            f.write("prob_threshold = " + str(prob_threshold) + "\n")
                                            f.write("lag = " + str(lag) + "\n")
                                            f.write("tree_depth = " + str(depth) + "\n")
                                            f.write("mchild_weight = " + str(min_child_weight) + "\n")
                                            f.write("eta = " + str(eta) + "\n")
                                        with open(write_to_file, 'a') as f:
                                            dfAsString = res_df.to_string(header=True, index=False)
                                            f.write(dfAsString)
                                            f.write("\n\n")

                                if profit > max_prof:
                                    max_prof = profit
                                    odds_res = odds_threshold
                                    prob_res = prob_threshold
                                    lag_res = lag
                                    max_depth_res = depth
                                    mchild_w_res = min_child_weight
                                    eta_res = eta
        return max_prof, odds_res, prob_res, lag_res, max_depth_res, mchild_w_res, eta_res

    def play_the_next_games(self, predicted_l, predicted_prob_l, X_test, odds_thres, bet, threshold, verbose):
        games_played = 0

        for i in range(X_test.shape[0]):

            pred = predicted_l[i]
            pred_prob = predicted_prob_l[i]

            if (pred == 1) & (pred_prob[1] >= threshold):
                if (X_test.iloc[i]['B365A'] > odds_thres) & (X_test.iloc[i]['B365D'] > odds_thres):
                    odds_sum = X_test.iloc[i]['B365A'] + X_test.iloc[i]['B365D']
                    ratio_draw = X_test.iloc[i]['B365A'] / odds_sum
                    ratio_away = X_test.iloc[i]['B365D'] / odds_sum

                    if verbose:
                        print('odds away:', X_test.iloc[i]['B365A'], 'odds draw:', X_test.iloc[i]['B365D'])
                        print('ratio draw:', bet * ratio_draw, 'ratio away:', bet * ratio_away)
                        print(X_test.iloc[i][['pl_game']])

                    games_played = games_played + 1

            if (pred == 0) & (pred_prob[0] > threshold):
                if X_test.iloc[i]['B365H'] > odds_thres:
                    if verbose:
                        print('odds home:', X_test.iloc[i]['B365H'])
                        print(X_test.iloc[i][['pl_game']])

                    games_played = games_played + 1

        return games_played

