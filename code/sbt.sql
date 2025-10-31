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
        , extub: CASE
            WHEN LAG(device_category) OVER w = 'imv'
                AND device_category != 'imv' THEN 1 ELSE 0 END
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
),

-- Use the gaps-and-islands technique to identify contiguous blocks of TRUE SBT states.
sbt_block_changes AS (
    FROM sbt_state
    SELECT *
        -- A new block starts when '_sbt_state' flips from FALSE to TRUE or TRUE to FALSE
        , _chg: CASE
            WHEN _sbt_state IS DISTINCT FROM LAG(_sbt_state) OVER w
            THEN 1 ELSE 0 END
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
),

-- Assign a unique ID to each SBT block
sbt_block_ids AS (
    FROM sbt_block_changes
    SELECT *
        -- The cumulative sum of the start flags creates a unique ID for each block
        , _block_id: SUM(_chg) OVER w
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
),

-- CTE 4: Calculate duration for each valid SBT block and check the mode that preceded it.
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
        , extub
)

FROM final_joined
--WHERE hospitalization_id IN ('20001361', '20004088', '20005024', '20006409')
ORDER BY hospitalization_id, recorded_dttm;
