-- Define the true "SBT State" at each timestamp, where ALL criteria are met.
WITH t1 AS (
    FROM resp_p
    SELECT 
        device_category
        , device_name
        , mode_category
        , mode_name
        --, mode_cat_id
        , fio2_set
        , peep_set
        , pressure_support_set
        , tracheostomy
        --, _prev_mode: LAG(mode_category, 1, 'none') OVER w
        , hospitalization_id, recorded_dttm
        , _sbt_state: CASE
            WHEN (mode_category IN ('pressure support/cpap') AND peep_set <= 8 AND pressure_support_set <= 8)
                OR regexp_matches(device_name, 't1[\s_-]?piece') 
                THEN 1 ELSE 0 END
        -- measure every time intubation occurs (defined by switching from device_category of anything else to 'imv')
        , _intub: CASE
            WHEN LAG(device_category) OVER w IS DISTINCT FROM 'imv'
                AND device_category = 'imv' THEN 1 ELSE 0 END
        -- measure each extubation event (transition from 'imv' to anything else)
        , _extub: CASE
            WHEN LAG(device_category) OVER w = 'imv'
                AND device_category IS DISTINCT FROM 'imv'
            THEN 1 ELSE 0 END
        , _trach_flip_to_1: CASE
            WHEN LAG(tracheostomy) OVER w = 0
                AND tracheostomy = 1 THEN 1 ELSE 0 END
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
)

-- Use the gaps-and-islands technique to identify contiguous blocks of TRUE SBT states.
, t2 AS (
    FROM t1
    SELECT *
        -- A new block starts when '_sbt_state' flips from FALSE to TRUE or TRUE to FALSE
        , _chg_sbt_state: CASE
            WHEN _sbt_state IS DISTINCT FROM LAG(_sbt_state) OVER w
            THEN 1 ELSE 0 END
        , _extub_cum: SUM(_extub) OVER w
        , _extub_1st: CASE
            WHEN _extub = 1 AND _extub_cum = 1 THEN 1 ELSE 0 END
        , _trach_flip_cum: SUM(_trach_flip_to_1) OVER w
        , _trach_1st: CASE
            WHEN _trach_flip_to_1 = 1 AND _trach_flip_cum = 1 THEN 1 ELSE 0 END
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
)

-- Assign a unique ID to each SBT block
, t3 AS (
    FROM t2
    SELECT *
        -- The cumulative sum of the start flags creates a unique ID for each block
        , _block_id: SUM(_chg_sbt_state) OVER w
        -- failed extubation is defined as reintubation within 24 hours
        , _fail_extub: CASE
            WHEN t2._extub_1st = 1 AND EXISTS (
                SELECT 1
                FROM t1
                WHERE t1.hospitalization_id = t2.hospitalization_id
                  AND t1._intub = 1
                  AND t1.recorded_dttm > t2.recorded_dttm
                  AND t1.recorded_dttm <= t2.recorded_dttm + INTERVAL 24 HOUR
            ) THEN 1 ELSE 0 END
        , _last_vitals_within_24h_of_extub: CASE
            WHEN t2._extub_1st = 1 AND EXISTS (
                SELECT 1
                FROM last_vitals_df lv
                WHERE lv.hospitalization_id = t2.hospitalization_id
                  AND lv.recorded_dttm >= t2.recorded_dttm
                  AND lv.recorded_dttm <= t2.recorded_dttm + INTERVAL 24 HOUR
            ) THEN 1 ELSE 0 END
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
)

-- Calculate duration for each valid SBT block and check the mode that preceded it.
, all_blocks AS (
    FROM t3
    SELECT hospitalization_id
        , _block_id
        , _sbt_state
        , _start_mode: ANY_VALUE(mode_category ORDER BY recorded_dttm)
        , _end_mode: ANY_VALUE(mode_category ORDER BY recorded_dttm)
        , _start_dttm: MIN(recorded_dttm)
        , _last_dttm: MAX(recorded_dttm)
        -- Get the mode category from the row immediately preceding the start of the block
    --WHERE _sbt_state -- This is crucial, we only analyze the actual SBT blocks
    GROUP BY hospitalization_id, _block_id, _sbt_state
)

, all_blocks_with_duration AS (
    FROM all_blocks
    SELECT *
        , _next_start_dttm: LEAD(_start_dttm) OVER w
        , _end_dttm: COALESCE(_next_start_dttm, _last_dttm)
        , _duration_mins: date_diff('minute', _start_dttm, _end_dttm)
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY _block_id)
)

-- Final Step: Join the analysis back to the original data and apply final logic
, t4 AS (
    FROM t3
    LEFT JOIN all_blocks_with_duration AS b
        ON t3.hospitalization_id = b.hospitalization_id
        AND t3._block_id = b._block_id
        -- AND t3._sbt_state
    ASOF LEFT JOIN cs_df as c
        ON -- t3._extub_1st = 1 AND -- find the code status most recent to the extubation (of which only the first one is relevant) 
        c.hospitalization_id = t3.hospitalization_id
        AND c.start_dttm <= t3.recorded_dttm
    ASOF LEFT JOIN hosp_df as h -- find the discharge category
        ON -- t3._extub_1st = 1 AND 
        t3.hospitalization_id = h.hospitalization_id
        AND t3.recorded_dttm <= h.discharge_dttm
    SELECT t3.fio2_set
        , t3.peep_set
        , t3.pressure_support_set
        , t3.tracheostomy
        , _block_duration_mins: COALESCE(b._duration_mins, 0)
        , t3.device_category, t3.device_name, t3.mode_category, t3.mode_name
        , t3.hospitalization_id, event_dttm: t3.recorded_dttm
        -- Final SBT flag: TRUE if the block duration is >= 30 mins 
        , sbt_done: CASE
            WHEN _block_duration_mins >= 30 AND t3._sbt_state = 1
            THEN 1 ELSE 0 END
        , _extub_1st
        , _intub
        , t3._trach_1st
        , _fail_extub
        , c.code_status_category
        , cs_start_dttm: c.start_dttm
        , h.discharge_category
        , discharge_dttm: h.discharge_dttm
        , _last_vitals_within_24h_of_extub
        , _withdrawl_lst: CASE
            WHEN _extub_1st = 1 
            AND TRIM(LOWER(code_status_category)) != 'full'
            AND TRIM(LOWER(discharge_category)) in ('hospice', 'expired')
            THEN 1 ELSE 0 END
        , _success_extub: CASE
            WHEN _extub_1st = 1
            AND _withdrawl_lst = 0
            AND _fail_extub = 0
            THEN 1 ELSE 0 END
        , _death_after_extub_wo_reintub: CASE
            WHEN _extub_1st = 1
            AND _last_vitals_within_24h_of_extub = 1
            AND _fail_extub = 0
            AND TRIM(LOWER(discharge_category)) in ('hospice', 'expired')
            -- AND _withdrawl_lst = 0
            THEN 1 ELSE 0 END
)

FROM t4
WHERE (tracheostomy = 0 OR _trach_1st = 1)
    AND hospitalization_id IN ('20001361', '20004088', '20005024', '20006409', '21341369', '20134240', '20008807', '20014600')
ORDER BY hospitalization_id, event_dttm;



-- WITH base AS (
--     FROM resp_p
--     INNER JOIN cohort_hosp_ids_df USING (hospitalization_id)
--     SELECT hospitalization_id
--          , recorded_dttm
--          , device_category, device_name
--          , mode_category, mode_name
--          , mode_cat_id
--          , fio2_set
--          , peep_set
--          , pressure_support_set
--          , tracheostomy
--          , _mode_prev: LAG(mode_category) OVER w_mode
--          , _switch_to_ps_cpap: CASE
--              WHEN contains(_mode_prev, 'control')
--               AND mode_category IN ('pressure support/cpap')
--              THEN 1 ELSE 0 END
--     WHERE hospitalization_id IN ('20001361', '20004088', '20005024')
--     WINDOW w_mode AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
-- ),
-- -- segment attributes keyed by existing mode_cat_id
-- seg_bounds AS (
--     FROM base
--     SELECT hospitalization_id
--          , _seg_id: mode_cat_id 
--          , _seg_start: MIN(recorded_dttm)
--          , _seg_last_seen: MAX(recorded_dttm)
--          , _seg_mode: ANY_VALUE(mode_category)
--     GROUP BY hospitalization_id, _seg_id
-- ),
-- -- compute segment end and previous segment mode
-- seg_timing AS (
--     FROM seg_bounds
--     SELECT hospitalization_id
--          , _seg_id
--          , _seg_mode
--          , _seg_start
--          , _seg_last_seen -- timing of the last row for this segment
--          , _next_seg_start: LEAD(_seg_start) OVER w_seg -- timing of the first row of the next segment
--          , _seg_end: COALESCE(_next_seg_start, _seg_last_seen)
--          , _prev_seg_mode: LAG(_seg_mode) OVER w_seg
--     WINDOW w_seg AS (PARTITION BY hospitalization_id ORDER BY _seg_start)
-- ),
-- seg_enriched AS (
--     FROM seg_timing
--     SELECT hospitalization_id
--          , _seg_id
--          , _seg_mode
--          , _seg_start
--          , _seg_end
--          , _prev_seg_mode
--          , _seg_duration_min: date_diff('minute', _seg_start, _seg_end)
--          , _is_ps_cpap_after_control: CASE
--              WHEN _seg_mode = 'pressure support/cpap'
--               AND contains(_prev_seg_mode, 'control')
--              THEN 1 ELSE 0 END
--          , _persists_30min: CASE
--              WHEN _seg_mode = 'pressure support/cpap'
--               AND contains(_prev_seg_mode, 'control')
--               AND date_diff('minute', _seg_start, _seg_end) >= 30
--              THEN 1 ELSE 0 END
-- ),
-- final AS (
--     FROM base AS t3
--     LEFT JOIN seg_enriched AS se
--       ON se.hospitalization_id = t3.hospitalization_id
--      AND se._seg_id           = t3.mode_cat_id
--     SELECT t3.hospitalization_id
--          , t3.recorded_dttm
--          , t3.device_category, t3.device_name
--          , t3.mode_category, t3.mode_name
--          , t3.fio2_set
--          , t3.peep_set
--          , t3.pressure_support_set
--          , t3.tracheostomy
--          , t3._switch_to_ps_cpap
--          , se._is_ps_cpap_after_control
--          , se._persists_30min
--          , _sbt_done: CASE 
--              WHEN se._persists_30min = 1 
--               AND t3.peep_set <= 8
--               AND t3.pressure_support_set <= 8
--              THEN 1 ELSE 0 END
-- )
-- SELECT *
-- FROM seg_enriched
-- ORDER BY hospitalization_id, _seg_id