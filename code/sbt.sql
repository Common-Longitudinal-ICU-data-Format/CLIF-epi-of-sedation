-- Define the true "SBT State" at each timestamp, where ALL criteria are met.
WITH sbt_state AS (
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
                OR regexp_matches(device_name, 't[\s_-]?piece') 
                THEN 1 ELSE 0 END
        -- measure every time intubation occurs (defined by switching from device_category of anything else to 'imv')
        , _intub: CASE
            WHEN LAG(device_category) OVER w IS DISTINCT FROM 'imv'
                AND device_category = 'imv' THEN 1 ELSE 0 END
        -- measure the first extubation (defined by switching from 'imv' to anything else)
        , _extub_1st: CASE
            WHEN LAG(device_category) OVER w = 'imv'
                AND device_category IS DISTINCT FROM 'imv' 
                AND recorded_dttm = MIN(recorded_dttm) OVER w
                THEN 1 ELSE 0 END
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
),

-- Use the gaps-and-islands technique to identify contiguous blocks of TRUE SBT states.
sbt_block_changes AS (
    FROM sbt_state
    SELECT *
        -- A new block starts when '_sbt_state' flips from FALSE to TRUE or TRUE to FALSE
        , _chg_sbt_state: CASE
            WHEN _sbt_state IS DISTINCT FROM LAG(_sbt_state) OVER w
            THEN 1 ELSE 0 END
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
),

-- Assign a unique ID to each SBT block
sbt_block_ids AS (
    FROM sbt_block_changes
    SELECT *
        -- The cumulative sum of the start flags creates a unique ID for each block
        , _block_id: SUM(_chg_sbt_state) OVER w
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
),

-- Calculate duration for each valid SBT block and check the mode that preceded it.
all_blocks AS (
    FROM sbt_block_ids
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
),

all_blocks_with_duration AS (
    FROM all_blocks
    SELECT *
        , _next_start_dttm: LEAD(_start_dttm) OVER w
        , _end_dttm: COALESCE(_next_start_dttm, _last_dttm)
        , _duration_mins: date_diff('minute', _start_dttm, _end_dttm)
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY _block_id)
),

-- Final Step: Join the analysis back to the original data and apply final logic
final_joined AS (
    FROM sbt_block_ids AS s
    LEFT JOIN all_blocks_with_duration AS b
        ON s.hospitalization_id = b.hospitalization_id
        AND s._block_id = b._block_id
        -- AND s._sbt_state
    SELECT s.fio2_set
        , s.peep_set
        , s.pressure_support_set
        , s.tracheostomy
        , _block_duration_mins: COALESCE(b._duration_mins, 0)
        , s.device_category, s.device_name, s.mode_category, s.mode_name
        , s.hospitalization_id, s.recorded_dttm
        -- Final SBT flag: TRUE if the block duration is >= 30 mins 
        , sbt_done: CASE
            WHEN _block_duration_mins >= 30 AND s._sbt_state = 1
            THEN 1 ELSE 0 END
        , _extub_1st
        , _intub
)

FROM final_joined
--WHERE hospitalization_id IN ('20001361', '20004088', '20005024', '20006409')
ORDER BY hospitalization_id, recorded_dttm;

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
--     FROM base AS s
--     LEFT JOIN seg_enriched AS se
--       ON se.hospitalization_id = s.hospitalization_id
--      AND se._seg_id           = s.mode_cat_id
--     SELECT s.hospitalization_id
--          , s.recorded_dttm
--          , s.device_category, s.device_name
--          , s.mode_category, s.mode_name
--          , s.fio2_set
--          , s.peep_set
--          , s.pressure_support_set
--          , s.tracheostomy
--          , s._switch_to_ps_cpap
--          , se._is_ps_cpap_after_control
--          , se._persists_30min
--          , _sbt_done: CASE 
--              WHEN se._persists_30min = 1 
--               AND s.peep_set <= 8
--               AND s.pressure_support_set <= 8
--              THEN 1 ELSE 0 END
-- )
-- SELECT *
-- FROM seg_enriched
-- ORDER BY hospitalization_id, _seg_id