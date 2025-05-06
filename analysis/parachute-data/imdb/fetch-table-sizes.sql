COPY (
    SELECT 
        '{' || string_agg('"' || table_name || '": ' || row_count, ', ') || '}' AS json_output
    FROM (
        SELECT 'aka_name' AS table_name, COUNT(*) AS row_count FROM aka_name
        UNION ALL SELECT 'cast_info', COUNT(*) FROM cast_info
        UNION ALL SELECT 'company_name', COUNT(*) FROM company_name
        UNION ALL SELECT 'comp_cast_type', COUNT(*) FROM comp_cast_type
        UNION ALL SELECT 'info_type', COUNT(*) FROM info_type
        UNION ALL SELECT 'kind_type', COUNT(*) FROM kind_type
        UNION ALL SELECT 'movie_companies', COUNT(*) FROM movie_companies
        UNION ALL SELECT 'movie_info_idx', COUNT(*) FROM movie_info_idx
        UNION ALL SELECT 'movie_link', COUNT(*) FROM movie_link
        UNION ALL SELECT 'person_info', COUNT(*) FROM person_info
        UNION ALL SELECT 'aka_title', COUNT(*) FROM aka_title
        UNION ALL SELECT 'char_name', COUNT(*) FROM char_name
        UNION ALL SELECT 'company_type', COUNT(*) FROM company_type
        UNION ALL SELECT 'complete_cast', COUNT(*) FROM complete_cast
        UNION ALL SELECT 'keyword', COUNT(*) FROM keyword
        UNION ALL SELECT 'link_type', COUNT(*) FROM link_type
        UNION ALL SELECT 'movie_info', COUNT(*) FROM movie_info
        UNION ALL SELECT 'movie_keyword', COUNT(*) FROM movie_keyword
        UNION ALL SELECT 'name', COUNT(*) FROM name
        UNION ALL SELECT 'role_type', COUNT(*) FROM role_type
        UNION ALL SELECT 'title', COUNT(*) FROM title
    )
) TO 'data-table-sizes.json' (FORMAT CSV, HEADER FALSE, DELIMITER '\n', QUOTE '');