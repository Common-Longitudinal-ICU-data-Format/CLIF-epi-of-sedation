WITH t1 AS (
    FROM sed_wg g
    SELECT hospitalization_id, event_dttm, _dh, _hr
        , LAST_VALUE(COLUMNS('_min') IGNORE NULLS) OVER (
            PARTITION BY hospitalization_id ORDER BY event_dttm
        )
        , _duration: EXTRACT(EPOCH FROM (LEAD(event_dttm, 1, event_dttm) OVER w - event_dttm)) / 60.0
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
), t2 AS (
    FROM t1
    SELECT hospitalization_id, event_dttm, _dh, _hr, _duration
        --, COALESCE(_duration_mins, 0)
        , COALESCE(COLUMNS('_min'), 0) 
), t3 AS (
    FROM t2
    SELECT hospitalization_id, event_dttm, _dh, _hr, _duration
        , COLUMNS('_min') * _duration
), t4 AS (
    FROM t3
    SELECT hospitalization_id, _dh, _hr
        , SUM(COLUMNS('_min'))
    GROUP BY hospitalization_id, _dh, _hr
)
SELECT *
FROM t4
-- ORDER BY hospitalization_id, event_dttm
ORDER BY hospitalization_id, _dh