from flask import Flask, request, jsonify
import datetime
import duckdb
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/send_transactions', methods=['POST'])
def send_transactions():
    transactions = request.json
    
    df = pd.json_normalize(eval(transactions)) 

    del transactions

    model_data = duckdb.query("""
                    SELECT df.*, tt.total
                      , (approved/total)*100 as approved_pct
                      , (failed/total)*100 as failed_pct
                      , (reversed/total)*100 as reversed_pct
                      , (denied/total)*100 as denied_pct
                      , (denied/(approved+0.1)) as denied_approved
                      , (failed/(approved+0.1)) as failed_approved
                      , (reversed/(approved+0.1)) as reversed_approved
                      , coalesce( (failed - approved)/((failed+approved)/2),0 )as failed_pct_diff
                      , coalesce( (denied - approved)/((denied+approved)/2),0 ) as denied_pct_diff
                      , coalesce( (reversed - approved)/((reversed+approved)/2),0 ) as reversed_pct_diff
                      , dummies.* EXCLUDE (time)
                    FROM df
                    LEFT JOIN (select time, (approved+denied+failed+reversed+processing+refunded) as total
                                from df) as tt                     
                    ON tt.time = df.time
                    LEFT JOIN (select time
                      ,IF ( CAST(df.time as STRING)[:2] == '00',1,0) as hour_00
                      ,IF ( CAST(df.time as STRING)[:2] == '01',1,0) as hour_01
                      ,IF ( CAST(df.time as STRING)[:2] == '02',1,0) as hour_02
                      ,IF ( CAST(df.time as STRING)[:2] == '03',1,0) as hour_03
                      ,IF ( CAST(df.time as STRING)[:2] == '04',1,0) as hour_04
                      ,IF ( CAST(df.time as STRING)[:2] == '05',1,0) as hour_05
                      ,IF ( CAST(df.time as STRING)[:2] == '06',1,0) as hour_06
                      ,IF ( CAST(df.time as STRING)[:2] == '07',1,0) as hour_07
                      ,IF ( CAST(df.time as STRING)[:2] == '08',1,0) as hour_08
                      ,IF ( CAST(df.time as STRING)[:2] == '09',1,0) as hour_09
                      ,IF ( CAST(df.time as STRING)[:2] == '10',1,0) as hour_10
                      ,IF ( CAST(df.time as STRING)[:2] == '11',1,0) as hour_11
                      ,IF ( CAST(df.time as STRING)[:2] == '12',1,0) as hour_12
                      ,IF ( CAST(df.time as STRING)[:2] == '13',1,0) as hour_13
                      ,IF ( CAST(df.time as STRING)[:2] == '14',1,0) as hour_14
                      ,IF ( CAST(df.time as STRING)[:2] == '15',1,0) as hour_15
                      ,IF ( CAST(df.time as STRING)[:2] == '16',1,0) as hour_16
                      ,IF ( CAST(df.time as STRING)[:2] == '17',1,0) as hour_17
                      ,IF ( CAST(df.time as STRING)[:2] == '18',1,0) as hour_18
                      ,IF ( CAST(df.time as STRING)[:2] == '19',1,0) as hour_19
                      ,IF ( CAST(df.time as STRING)[:2] == '20',1,0) as hour_20
                      ,IF ( CAST(df.time as STRING)[:2] == '21',1,0) as hour_21
                      ,IF ( CAST(df.time as STRING)[:2] == '22',1,0) as hour_22
                      ,IF ( CAST(df.time as STRING)[:2] == '23',1,0) as hour_23
                        from df
                        group by time) as dummies
                    ON dummies.time = df.time
    """).df() 

    # Passing copy of data, remove data not used by models
    X = model_data.drop(['time','total'],axis=1)


    # Calculate distance from correlation denied X approved
    clf_linear = joblib.load('Models\\linearR_denied.save')
    x_approved = np.array(X.approved).reshape(-1,1)
    X['denied_sqr_distance'] = (X.denied - clf_linear.predict(x_approved))**2

    # Ordering columns from moment of fit
    X = X[['approved', 'denied', 'failed', 'processing', 'refunded', 'reversed',
       'denied_approved', 'denied_pct', 'reversed_approved', 'reversed_pct',
       'failed_approved', 'failed_pct', 'failed_pct_diff', 'denied_pct_diff',
       'reversed_pct_diff', 'approved_pct', 'denied_sqr_distance', 'hour_00',
       'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05', 'hour_06',
       'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23']]
    

    # Load Models
    clf_failed = joblib.load('Models\\clf_failed.save')
    clf_reversed = joblib.load('Models\\clf_reversed.save')
    clf_denied = joblib.load('Models\\clf_denied.save')
    scaler = joblib.load('Models\\scaler.save')


    # Pre processing
    X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    X_failed = X[['failed', 'failed_approved', 'failed_pct', 'failed_pct_diff', 'denied_pct_diff',
                   'reversed_pct_diff', 'hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04',
                     'hour_05', 'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                       'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
                         'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23']]
    
    X_denied = X[['approved', 'denied', 'denied_pct', 'approved_pct',
                   'denied_sqr_distance', 'hour_00', 'hour_01', 'hour_02', 'hour_03',
                     'hour_04', 'hour_05', 'hour_06', 'hour_07', 'hour_08', 'hour_09',
                       'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15',
                         'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
                           'hour_22', 'hour_23']]
    
    X_reversed = X[['reversed', 'reversed_pct', 'reversed_pct_diff', 'approved_pct',
                         'hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
                           'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                             'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
                               'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23']]
    
    # Predictions and scores
    score_failed = clf_failed.score_samples(X_failed) *-1
    score_denied = clf_denied.score_samples(X_denied) *-1
    score_reversed = clf_reversed.score_samples(X_reversed) *-1


    df['anomaly_failed'] = score_failed > 0.48 #or df.failed >= 1
    df['anomaly_denied'] = score_denied > 0.43
    df['anomaly_reversed'] = score_reversed > 0.44

    return jsonify(df.to_dict('records'))
    
    

app.run()