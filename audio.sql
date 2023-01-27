BEGIN TRANSACTION;
CREATE TABLE AUDIO
             (ID INT PRIMARY KEY     NOT NULL,
             CHANNELS           INT    NOT NULL,
             FRAMERATE          INT     NOT NULL,
             NFRAMES            INT     NOT NULL);
CREATE TABLE AUDIO_PARAM 
             (ID INT PRIMARY KEY     NOT NULL,
                min_sig                  INT    NOT NULL,
                max_sig                  INT     NOT NULL,
                mean_sig                 INT     NOT NULL,
                min_mean                 INT     NOT NULL,
                max_mean                 INT     NOT NULL,
                centroid                 INT     NOT NULL,
                RMS_sig                  INT     NOT NULL,
                std_sig                  INT     NOT NULL,
                mean_skewness            INT     NOT NULL,
                mean_kurtosis            INT     NOT NULL,
                skewness                 INT     NOT NULL,
                kurtosis                 INT     NOT NULL,
                shannon                  INT     NOT NULL,
                renyi                    INT     NOT NULL,
                rate_attack              INT     NOT NULL,
                rate_decay               INT     NOT NULL,
                silence_ratio            INT     NOT NULL,
                threshold_crossing_rate  INT     NOT NULL);
COMMIT;
