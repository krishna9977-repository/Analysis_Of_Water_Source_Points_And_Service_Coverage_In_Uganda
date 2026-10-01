/*********************************************
TAB PURPOSE: EDA (EXPLORATORY DATA ANALYSIS) and DATA CLEANING QUERIES
DATASET: waterpoint
AUTHOR: KRISHNA SHETE
********************************************/
-- select * from waterpoint;
-- DESCRIBE waterpoint; = data types of all the column is 'text' AND all the columns contain 'null' values 
-- SHOW TABLE STATUS LIKE 'waterpoint';


/*****
select count(*) as total_columns
from information_schema.columns
where table_schema = 'waterpointdata'
and table_name = 'waterpoint';
*********/

/*********
select count(*) as total_rows
from waterpoint;
= 832639
*********/

/*************
select row_id from waterpoint
where row_id is null
or row_id = ''
or row_id regexp '[^0-9]';   = returned all the columns because the field type in the column is 'text' as of now
***************/

/*********
select row_id from waterpoint
where row_id is null;
************/

/*********
select count(*) as empty_or_space
from waterpoint
where trim(row_id) = ''; = no empty string, no space-only values, every row id has visible value
**********/

/**********
SELECT COUNT(*) AS non_numeric
FROM waterpoint
WHERE row_id REGEXP '[^0-9]';
***************/

/*************
SELECT COUNT(*) AS non_numeric_rows
FROM waterpoint
WHERE row_id IS NULL
   OR TRIM(row_id) = ''
   OR row_id REGEXP '[^0-9]';
**********/

/*************
SELECT COUNT(*) AS bad_rows
FROM waterpoint
WHERE row_id IS NULL
   OR TRIM(row_id) = ''
   OR CAST(row_id AS UNSIGNED) = 0 AND row_id <> '0'; = 1 row found
***************/

/****************
SELECT *
FROM waterpoint
WHERE row_id IS NULL
   OR TRIM(row_id) = ''
   OR CAST(row_id AS UNSIGNED) = 0 AND row_id <> '0';
*****************/


/***************
UPDATE waterpoint
SET row_id = NULL
WHERE row_id = 'SP 14 - 13'; = it didn't worked failed to execute
***************/

-- set sql_safe_updates = 0;

/************
UPDATE waterpoint
SET row_id = null
WHERE row_id = 'SP 14 - 13';
**************/

-- set sql_safe_updates = 1;


/*********
select count(*) as bad_rows
from waterpoint
where row_id is not null
and row_id regexp '[^0-9]';
**************/

/***************
SELECT DISTINCT count row_id
FROM waterpoint
WHERE row_id IS NOT NULL
  AND row_id REGEXP '[^0-9]';
****************/

/*****************
SELECT DISTINCT row_id
FROM waterpoint
WHERE row_id LIKE '%,%';
*****************/

-- set sql_safe_updates = 0;

/**************
UPDATE waterpoint
SET row_id = REPLACE(row_id, ',', '')
WHERE row_id LIKE '%,%';   = converted eg. "1,234" to "1234"
******************/

-- set sql_safe_updates = 1;


/***********
UPDATE waterpoint
SET row_id = TRIM(row_id);
*******/

-- set sql_safe_updates = 0;

/************
UPDATE waterpoint
SET row_id = TRIM(row_id);
*************/


-- set sql_safe_updates = 1;

/*********
SELECT COUNT(*) AS bad_rows
FROM waterpoint
WHERE row_id IS NOT NULL
  AND row_id REGEXP '[^0-9]';   = now it returned 0
************/

/***********
ALTER TABLE waterpoint
MODIFY COLUMN row_id INT;  = converted the field type of row_id from "text" to "int", null stayed null , no rows were rejected, 
**************/

-- describe waterpoint;

/*********
SELECT row_id, COUNT(*)
FROM waterpoint
GROUP BY row_id
HAVING COUNT(*) > 1;   =  no count was returned which means all the rows are unique
*************/


/**********
select * from waterpoint
order by row_id asc;
***********/

/***************
ALTER TABLE waterpoint
ADD PRIMARY KEY (row_id);  = didn't converted to primary key because there's a null value present
*************/



/*************
select row_id from waterpoint
where row_id is null
***********/


/************
alter table waterpoint
add column id int auto_increment first;    = didn't worked it said there can only one auto column and it must be defined as a key
*************/


/*************
ALTER TABLE waterpoint
ADD COLUMN id INT AUTO_INCREMENT,
ADD UNIQUE KEY (id);
**************/

/***************
SELECT id, row_id
FROM waterpoint
WHERE row_id IS NULL
ORDER BY id;
*******************/

/**********
SELECT *
FROM waterpoint
WHERE id BETWEEN
      (SELECT MIN(id) - 2 FROM waterpoint WHERE row_id IS NULL)
  AND (SELECT MAX(id) + 2 FROM waterpoint WHERE row_id IS NULL)
ORDER BY id;															= gave rows having null values in row_id column and few rows below it and few rows below it
***************/


/****************
WITH ordered AS (
  SELECT *,
         ROW_NUMBER() OVER (
           ORDER BY row_id IS NULL, row_id
         ) AS rn
  FROM waterpoint
)
SELECT *
FROM ordered
WHERE rn BETWEEN
      (SELECT rn - 3 FROM ordered WHERE row_id IS NULL LIMIT 1)
  AND (SELECT rn + 3 FROM ordered WHERE row_id IS NULL LIMIT 1)
ORDER BY rn;
*******************/



/************
SELECT *
FROM waterpoint
where row_id is null;
***************/


-- set sql_safe_updates = 0;

/**************
delete from waterpoint
where row_id is null;
***************/


-- set sql_safe_updates = 1;


/*****************
SELECT *
FROM waterpoint
where row_id is null; = the row which was having null value in row_id has been deleted
******************/

/***************
ALTER TABLE waterpoint
DROP COLUMN id;
*************/


-- select * from waterpoint;


/***************
SELECT row_id, COUNT(*) AS cnt
FROM waterpoint
WHERE row_id IS NOT NULL
GROUP BY row_id
HAVING COUNT(*) > 1;   		= no rows returned --- all rows are unique
**************/

-- select * from waterpoint;

/***************
ALTER TABLE waterpoint
ADD PRIMARY KEY (row_id);		=	row_id added as primary key for the waterpoint table
**************/

/************
SELECT COUNT(*) AS null_count
FROM waterpoint
WHERE source IS NULL;  -- counts null values in column "source"
******************/


/********************
SELECT COUNT(*)
FROM waterpoint
WHERE TRIM(source) = '';   = Returned count from the source column where there are empty spaces, there were 5664 empty spaces.
********************/



/**************
SELECT *
FROM waterpoint
WHERE TRIM(source) = '';
****************/



/******************
SELECT * 
FROM waterpoint
WHERE TRIM(source) = '';
****************/


-- set sql_safe_updates = 0;


/*********************
UPDATE waterpoint
SET source = 'Unknown'
WHERE TRIM(source) = '';
*********************/


-- set sql_safe_updates = 1;


/**********************
SELECT * 
FROM waterpoint
WHERE TRIM(source) = ''; = returned 0 rows
*************************/

/*****************
SELECT
    source,
    COUNT(*) AS source_count
FROM waterpoint
GROUP BY source
ORDER BY source_count DESC;     = list all the unique in source (name of organization collecting the data record) and its count
*************************/ 


/***************
SELECT COUNT(*) AS null_count
FROM waterpoint
WHERE source IS NULL;		=	after running this query it resulted 0 null count, here I learned that the ( DESCRIBE waterpoint; ) when we execute this it says yes in the table under null colums - what it means is that even if there are no null values in score column it is allowed to keep null values so yes.
*******************/

/************
SELECT
  COUNT(*) AS null_lat
FROM waterpoint
WHERE lat_deg IS NULL;

SELECT
  COUNT(*) AS null_lon
FROM waterpoint
WHERE lon_deg IS NULL; 				= there are 0 null values in both latitude and longatitude column
***************/


/*************************
SELECT COUNT(*) AS empty_lat
FROM waterpoint
WHERE TRIM(lat_deg) = '';

SELECT COUNT(*) AS empty_lon
FROM waterpoint
WHERE TRIM(lon_deg) = '';			= there are no any empty spaces in either of the column
*******************************/ 


/******************
SELECT *
FROM waterpoint
WHERE lat_deg IS NOT NULL
  AND TRIM(lat_deg) <> ''
  AND lat_deg REGEXP '[^0-9.-]';

SELECT *
FROM waterpoint
WHERE lon_deg IS NOT NULL
  AND TRIM(lon_deg) <> ''
  AND lon_deg REGEXP '[^0-9.-]';  = even though it flagged some rows it is not wrong it is scientific notions
*************************/


/********************
SELECT COUNT(*) AS non_convertible_lat
FROM waterpoint
WHERE lat_deg IS NOT NULL
  AND TRIM(lat_deg) <> ''
  AND CAST(lat_deg AS DOUBLE) IS NULL;



SELECT COUNT(*) AS non_convertible_lon
FROM waterpoint
WHERE lon_deg IS NOT NULL
  AND TRIM(lon_deg) <> ''
  AND CAST(lon_deg AS DOUBLE) IS NULL;				= gave 0 for both
*************************************************/


/******************
ALTER TABLE waterpoint
MODIFY COLUMN lat_deg DOUBLE;

ALTER TABLE waterpoint
MODIFY COLUMN lon_deg DOUBLE;			= converted field type to DOUBLE, earlier it was TEXT
********************/



/*********************
SELECT COUNT(*) AS bad_report_dates
FROM waterpoint
WHERE report_date IS NOT NULL
  AND TRIM(report_date) <> ''
  AND STR_TO_DATE(report_date, '%Y %b %d %h:%i:%s %p') IS NULL; 		= every date is in this (2015 Jun 15 12:00:00 AM) format
*******************/



/***********************
SELECT COUNT(*) AS null_count
FROM waterpoint
WHERE report_date IS NULL;			= there are 0 null values
*************************/


/*********************
SELECT COUNT(*) AS empty_string_count
FROM waterpoint
WHERE TRIM(report_date) = '';				= there are zero empty spaces
*************************/

-- set sql_safe_updates = 0;

/***************
UPDATE waterpoint
SET report_date = STR_TO_DATE(report_date, '%Y %b %d %h:%i:%s %p')
WHERE report_date IS NOT NULL
  AND TRIM(report_date) <> '';
******************/

/****************
ALTER TABLE waterpoint
MODIFY COLUMN report_date DATETIME;
******************/


-- set sql_safe_updates = 0;


/************
ALTER TABLE waterpoint
MODIFY COLUMN report_date DATE;
*****************/

-- set sql_safe_updates = 1;


/***********
SELECT COUNT(DISTINCT status_id) AS unique_status_id_count
FROM waterpoint;
****************/

/**************
SELECT status_id, COUNT(*) AS count
FROM waterpoint
GROUP BY status_id
ORDER BY count DESC;
********************/



/***************
SELECT COUNT(*) AS null_count
FROM waterpoint
WHERE status_id IS NULL;
*****************/



/*************
SELECT COUNT(*) AS empty_or_space_count
FROM waterpoint
WHERE TRIM(status_id) = '';
******************/


-- WATER_SOURCE COLUMN

/******************
SELECT COUNT(*) AS null_count
FROM waterpoint
WHERE water_source IS NULL;			= 0 null values
****************/


/**********************
SELECT *
FROM waterpoint
WHERE water_source IS NOT NULL
  AND TRIM(water_source) = '';
**************************/


-- set sql_safe_updates = 0;



/******************
UPDATE waterpoint
SET water_source = 'unknown'
WHERE water_source IS NOT NULL
  AND TRIM(water_source) = '';
********************/




-- set sql_safe_updates = 1;



/********************
SELECT water_source, COUNT(*)
FROM waterpoint
GROUP BY water_source
ORDER BY COUNT(*) DESC;		= 87443 rows set to unknown
**********************/




/*************************
SELECT *
FROM waterpoint
WHERE water_source IS NOT NULL
  AND TRIM(water_source) = '';		= checks if still are there null values left --> No, null values left all were filled by unknown
  **************************/



-- WATER_TECH
/************************
SELECT COUNT(*) AS null_count
FROM waterpoint
WHERE water_tech IS NULL;			= No null values in water_tech column but empty spaces should be confirmed
**********************/



/***************
SELECT COUNT(*) AS empty_or_space_count
FROM waterpoint
WHERE water_tech IS NOT NULL
  AND TRIM(water_tech) = '';			= confirms empty spaces are there 398320
*********************/



/*******************
SELECT water_tech, COUNT(*) AS count
FROM waterpoint
GROUP BY water_tech
ORDER BY count DESC;
**********************/




-- set sql_safe_updates = 0;


/*****************
UPDATE waterpoint
SET water_tech = 'unknown'
WHERE water_tech IS NOT NULL
  AND TRIM(water_tech) = ''; 			= Updated all empty spaces (3998320) to unknown
*********************/


-- set sql_safe_updates = 1;

/*********************
SELECT water_tech, COUNT(*) AS count
FROM waterpoint
GROUP BY water_tech
ORDER BY water_tech;
*********************/


-- COUNTRY_NAME

/***************
SELECT country_name, COUNT(*) AS country_name_count
FROM waterpoint
GROUP BY country_name
ORDER BY country_name;
*******************/


/******************
SELECT COUNT(*) AS empty_or_space_count
FROM waterpoint
WHERE country_name IS NOT NULL
  AND TRIM(country_name) = '';		= empty spaces need to be filled with the country names if it will be possible to fill with the help of the values in columns latitude and longatitude, or not able to find then will be filled with unknown
*********************/


/******************
select row_id, lat_deg, lon_deg, country_name
from waterpoint
where country_name is not null
and trim(country_name) = '';
*********************/


/*******************
SELECT COUNT(*) AS missing_country_name_count
FROM waterpoint
WHERE country_name IS NULL
   OR TRIM(country_name) = ''; 		= the result is 139729
********************/


/*****************
SELECT row_id, lat_deg, lon_deg, country_name
FROM waterpoint
WHERE country_name IS NULL
   OR TRIM(country_name) = '';
**********************/

/*****************
SELECT adm1, COUNT(*) AS unique_adm1_count
FROM waterpoint
group by adm1
*******************/


/**********************
SELECT COUNT(*) AS null_count
FROM waterpoint
WHERE adm1 IS NULL;
*******************/



/********************
SELECT COUNT(*) AS empty_or_space_count
FROM waterpoint
WHERE adm1 IS NOT NULL
  AND TRIM(adm1) = '';
*****************/


/***************
SELECT adm1, COUNT(*) AS unique_adm1_count
FROM waterpoint
group by adm1
****************/



 -- set sql_safe_updates = 0;
 
 
 /*********************
 UPDATE waterpoint
SET adm1 = 'Unknown'
WHERE adm1 IS NOT NULL
  AND TRIM(adm1) = '';
***************************/

-- set sql_safe_updates = 1;



/*****************
SELECT adm1, COUNT(*) AS unique_adm1_count
FROM waterpoint
group by adm1
*******************/


/**************
SELECT MAX(CHAR_LENGTH(adm1)) AS max_length
FROM waterpoint; 			= 72
***************/

/************
ALTER TABLE waterpoint
MODIFY COLUMN adm1 VARCHAR(100);
*****************/


/********************
SELECT MAX(CHAR_LENGTH(country_name)) AS max_length
FROM waterpoint;			=	24
****************/



/*************
ALTER TABLE waterpoint
MODIFY COLUMN country_name VARCHAR(100);
******************/




/****************
ALTER TABLE waterpoint
MODIFY COLUMN status_id VARCHAR(10);
********************/



/*******************
SELECT *
FROM waterpoint
WHERE water_source IS NOT NULL
  AND TRIM(water_source) = '';	
******************/



/**********************
SELECT MAX(CHAR_LENGTH(water_source)) AS max_length
FROM waterpoint;			= maximun length is 123
***************/



/*******************
ALTER TABLE waterpoint
MODIFY COLUMN water_source VARCHAR(150);
**********************/


-- ADM2 COLUMN



/*************
SELECT COUNT(*) AS null_count
FROM waterpoint
WHERE adm2 IS NULL;			= 0 null values
*********************/



/*************************
SELECT COUNT(*) AS empty_or_space_count
FROM waterpoint
WHERE adm2 IS NOT NULL
  AND TRIM(adm2) = '';			= 171727 empty spaces precent in adm2 column
****************************/




/*****************
SELECT adm2, COUNT(*) AS cnt
FROM waterpoint
GROUP BY adm2
ORDER BY cnt DESC;
********************/



-- set sql_safe_updates = 0;


/****************
UPDATE waterpoint
SET adm2 = 'Unknown'
WHERE adm2 IS NOT NULL
  AND TRIM(adm2) = '';
**********************/


-- set sql_safe_updates = 1;


/****************
SELECT MAX(CHAR_LENGTH(adm2)) AS max_length
FROM waterpoint;
***************/



/*****************
ALTER TABLE waterpoint
MODIFY COLUMN adm2 VARCHAR(100);
********************/




-- adm3 column


/*******************
SELECT COUNT(*) AS null_count
FROM waterpoint
WHERE adm3 IS NULL;			= 0
******************/




/************************
SELECT COUNT(*) AS empty_or_space_count
FROM waterpoint
WHERE adm3 IS NOT NULL
  AND TRIM(adm3) = '';			= 744100 empty spaces, so the adm3 column will be droped
****************************/


-- set sql_safe_updates = 0;


/************************
ALTER TABLE waterpoint
DROP COLUMN adm3;		= adm3 column is droped because 744100 out of 832638 are unknown
****************/


-- INSTALL YEAR

/*****************
SELECT COUNT(*) AS null_count
FROM waterpoint
WHERE install_year IS NULL;			= 0
*******************/


/******************
SELECT COUNT(*) AS empty_or_space_count
FROM waterpoint
WHERE install_year IS NOT NULL
  AND TRIM(install_year) = '';				= has 321543 empty spaces
*************************/


/************
SELECT install_year, COUNT(*) AS cnt
FROM waterpoint
GROUP BY install_year
ORDER BY cnt DESC;					= has install_year from 1900 - 2050 but as we are in 2026, need to clean the years after 2026
*****************/



/**********************
SELECT install_year, COUNT(*) AS cnt
FROM waterpoint
GROUP BY install_year
ORDER BY cnt DESC;
*****************/



/******************
UPDATE waterpoint
SET install_year = 'Unknown'
WHERE install_year IS NOT NULL
  AND TRIM(install_year) = '';
***********************/




/********************
SELECT install_year, COUNT(*) AS cnt
FROM waterpoint
GROUP BY install_year
ORDER BY cnt DESC;
**********************/


/*******************
UPDATE waterpoint
SET install_year = NULL
WHERE install_year = 'unknown';
******************/

/*****************
SELECT install_year, COUNT(*) AS cnt
FROM waterpoint
GROUP BY install_year
ORDER BY cnt DESC;
**********************/

/****************
ALTER TABLE waterpoint
MODIFY COLUMN install_year INT;
***************/


/****************
SELECT
  SUM(installer IS NULL)      AS installer_nulls,
  SUM(rehab_year IS NULL)     AS rehab_year_nulls,
  SUM(rehabilitator IS NULL)  AS rehabilitator_nulls
FROM waterpoint;
*******************/


/******************
SELECT
  SUM(installer IS NOT NULL AND TRIM(installer) = '')      AS installer_empty_spaces,
  SUM(rehab_year IS NOT NULL AND TRIM(rehab_year) = '')     AS rehab_year_empty_spaces,
  SUM(rehabilitator IS NOT NULL AND TRIM(rehabilitator) = '') AS rehabilitator_empty_spaces
FROM waterpoint;
				= 	 ( installer has 735530, rehab_year has 831928, rehabilitator has 831561 ) empty spaces out of 832638 rows, so it's better to drop these columns
************************************/



-- set sql_safe_updates = 0;


/******************************
ALTER TABLE waterpoint
DROP COLUMN installer,
DROP COLUMN rehab_year,
DROP COLUMN rehabilitator;
********************************/


-- set sql_safe_updates = 1;



/**************************************
SELECT
  SUM(management IS NULL) AS management_nulls,
  SUM(pay IS NULL) AS pay_nulls,
  SUM(status IS NULL) AS status_nulls,
  SUM(fecal_coliform_value IS NULL) AS fecal_coliform_value_nulls,
  SUM(fecal_coliform_presence IS NULL) AS fecal_coliform_presence_nulls,
  SUM(subjective_quality IS NULL) AS subjective_quality_nulls,
  SUM(activity_id IS NULL) AS activity_id_nulls,
  SUM(scheme_id IS NULL) AS scheme_id_nulls,
  SUM(notes IS NULL) AS notes_nulls,
  SUM(photo_lnk IS NULL) AS photo_lnk_nulls,
  SUM(orig_lnk IS NULL) AS orig_lnk_nulls,
  SUM(data_lnk IS NULL) AS data_lnk_nulls,
  SUM(country_id IS NULL) AS country_id_nulls,
  SUM(clean_country_id IS NULL) AS clean_country_id_nulls,
  SUM(clean_country_name IS NULL) AS clean_country_name_nulls,
  SUM(clean_adm1 IS NULL) AS clean_adm1_nulls,
  SUM(clean_adm2 IS NULL) AS clean_adm2_nulls,
  SUM(clean_adm3 IS NULL) AS clean_adm3_nulls,
  SUM(clean_adm4 IS NULL) AS clean_adm4_nulls,
  SUM(water_source_clean IS NULL) AS water_source_clean_nulls,
  SUM(water_source_category IS NULL) AS water_source_category_nulls,
  SUM(water_tech_clean IS NULL) AS water_tech_clean_nulls,
  SUM(water_tech_category IS NULL) AS water_tech_category_nulls,
  SUM(facility_type IS NULL) AS facility_type_nulls,
  SUM(management_clean IS NULL) AS management_clean_nulls,
  SUM(status_clean IS NULL) AS status_clean_nulls,
  SUM(count IS NULL) AS count_nulls,
  SUM(converted IS NULL) AS converted_nulls,
  SUM(public_data_source IS NULL) AS public_data_source_nulls,
  SUM(created_timestamp IS NULL) AS created_timestamp_nulls,
  SUM(pay_clean IS NULL) AS pay_clean_nulls,
  SUM(subjective_quality_clean IS NULL) AS subjective_quality_clean_nulls,
  SUM(dataset_title IS NULL) AS dataset_title_nulls
FROM waterpoint;										= no null values in any of these columns
***************************/




/*********************************************
SELECT
  SUM(management IS NOT NULL AND TRIM(management) = '') AS management_empty,
  SUM(pay IS NOT NULL AND TRIM(pay) = '') AS pay_empty,
  SUM(status IS NOT NULL AND TRIM(status) = '') AS status_empty,
  SUM(subjective_quality IS NOT NULL AND TRIM(subjective_quality) = '') AS subjective_quality_empty,
  SUM(notes IS NOT NULL AND TRIM(notes) = '') AS notes_empty,
  SUM(photo_lnk IS NOT NULL AND TRIM(photo_lnk) = '') AS photo_lnk_empty,
  SUM(orig_lnk IS NOT NULL AND TRIM(orig_lnk) = '') AS orig_lnk_empty,
  SUM(data_lnk IS NOT NULL AND TRIM(data_lnk) = '') AS data_lnk_empty,
  SUM(clean_country_name IS NOT NULL AND TRIM(clean_country_name) = '') AS clean_country_name_empty,
  SUM(clean_adm1 IS NOT NULL AND TRIM(clean_adm1) = '') AS clean_adm1_empty,
  SUM(clean_adm2 IS NOT NULL AND TRIM(clean_adm2) = '') AS clean_adm2_empty,
  SUM(clean_adm3 IS NOT NULL AND TRIM(clean_adm3) = '') AS clean_adm3_empty,
  SUM(clean_adm4 IS NOT NULL AND TRIM(clean_adm4) = '') AS clean_adm4_empty,
  SUM(water_source_clean IS NOT NULL AND TRIM(water_source_clean) = '') AS water_source_clean_empty,
  SUM(water_source_category IS NOT NULL AND TRIM(water_source_category) = '') AS water_source_category_empty,
  SUM(water_tech_clean IS NOT NULL AND TRIM(water_tech_clean) = '') AS water_tech_clean_empty,
  SUM(water_tech_category IS NOT NULL AND TRIM(water_tech_category) = '') AS water_tech_category_empty,
  SUM(facility_type IS NOT NULL AND TRIM(facility_type) = '') AS facility_type_empty,
  SUM(management_clean IS NOT NULL AND TRIM(management_clean) = '') AS management_clean_empty,
  SUM(status_clean IS NOT NULL AND TRIM(status_clean) = '') AS status_clean_empty,
  SUM(public_data_source IS NOT NULL AND TRIM(public_data_source) = '') AS public_data_source_empty,
  SUM(dataset_title IS NOT NULL AND TRIM(dataset_title) = '') AS dataset_title_empty
FROM waterpoint;
***************************************/





/***********************
SELECT management, management_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY management, management_clean
ORDER BY cnt DESC;
***********************/

/***************************************
SELECT COUNT(*) AS rows_to_update
FROM waterpoint
WHERE
  management IS NOT NULL
  AND management_clean IS NOT NULL
  AND TRIM(management) = ''
  AND TRIM(management_clean) = '';			= 353372 rows have empty spaces in both the column
************************************/





-- set sql_safe_updates = 0;


/*************************
UPDATE waterpoint
SET
  management = 'Unknown',
  management_clean = 'Unknown'
WHERE
  management IS NOT NULL
  AND management_clean IS NOT NULL
  AND TRIM(management) = ''
  AND TRIM(management_clean) = '';
*************************************/



-- set sql_safe_updates = 1;

/********************************
SELECT management, management_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY management, management_clean
ORDER BY cnt DESC;
*************************/



/****************************************
SELECT management, COUNT(*) AS cnt
FROM waterpoint
GROUP BY management
ORDER BY cnt DESC;
****************************/



/****************************************
SELECT COUNT(*) AS rows_to_update
FROM waterpoint
WHERE
  management IS NOT NULL
  AND TRIM(management) = '';			= every row in management column has been filled with some value
  ***********************************/
  
  
  

/*****************************
SELECT management, management_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY management, management_clean
ORDER BY cnt DESC;
*******************************/




/*************************
SELECT management_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY management_clean
ORDER BY cnt DESC;					= only 2828 rows has empty spaces, so it has left untouched and asumed that it is same as the row in management column
***********************************/


/*************************************
ALTER TABLE waterpoint
MODIFY COLUMN management VARCHAR(255),
MODIFY COLUMN management_clean VARCHAR(255);
**************************************************/


/*****************************
ALTER TABLE waterpoint
MODIFY COLUMN management_clean VARCHAR(255)
AFTER management;						= moved management_clean column next to management
**********************************/



/*****************
SELECT pay, pay_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY pay, pay_clean
ORDER BY cnt DESC;	
*************************/


-- set sql_safe_updates = 0;


/*****************************
UPDATE waterpoint
SET
  pay = 'Unknown',
  pay_clean = 'Unknown'
WHERE
  pay IS NOT NULL
  AND pay_clean IS NOT NULL
  AND TRIM(pay) = ''
  AND TRIM(pay_clean) = '';
  ******************************/
  
  
-- set sql_safe_updates = 1;
  

/***************************  
SELECT pay, pay_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY pay, pay_clean
ORDER BY cnt DESC;	
******************************/


/*********************************
ALTER TABLE waterpoint
MODIFY COLUMN pay_clean TEXT
AFTER pay;
*********************************/



/*********************
SELECT status, status_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY status, status_clean
ORDER BY cnt DESC;	
**********************/



-- set sql_safe_updates = 0;



/******************************
UPDATE waterpoint
SET status = 'Non-Functional'
WHERE status IS NOT NULL
  AND TRIM(status) = '';
********************************/


-- set sql_safe_updates = 1;


/************************************
SELECT status, status_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY status, status_clean
ORDER BY cnt DESC;
***********************************/


/***************************
ALTER TABLE waterpoint
MODIFY COLUMN status VARCHAR(500),
MODIFY COLUMN status_clean VARCHAR(100);
********************************/


/************************************
ALTER TABLE waterpoint
MODIFY COLUMN status_clean VARCHAR(100)
AFTER status;
********************************/




/***********************************
SELECT fecal_coliform_value, fecal_coliform_presence, COUNT(*) AS cnt
FROM waterpoint
GROUP BY fecal_coliform_value, fecal_coliform_presence 				= as there are 805808 empty spaces in both the columns, many reasons for the empty ness can be 1. Lack of equipment to test the water, 2. Not able to provide the equipment to test the water, 3. 
ORDER BY cnt DESC;
****************************************/





-- set sql_safe_updates = 0;


/*****************************
UPDATE waterpoint
SET fecal_coliform_value = 'unknown'
WHERE fecal_coliform_value IS NOT NULL
  AND TRIM(fecal_coliform_value) = '';
********************************************/


/***********************************
UPDATE waterpoint
SET fecal_coliform_presence = 'unknown'
WHERE fecal_coliform_presence IS NOT NULL
  AND TRIM(fecal_coliform_presence) = '';
*********************************/


-- set sql_safe_updates = 1;


/*******************
SELECT subjective_quality, subjective_quality_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY subjective_quality, subjective_quality_clean
ORDER BY cnt DESC;
*************************/


/************************
ALTER TABLE waterpoint
MODIFY COLUMN subjective_quality_clean TEXT
AFTER subjective_quality;
****************************/




/******************************
SELECT COUNT(*) AS rows_to_update
FROM waterpoint
WHERE subjective_quality IS NOT NULL
  AND subjective_quality_clean IS NOT NULL
  AND TRIM(subjective_quality) = ''
  AND TRIM(subjective_quality_clean) = '';
**************************************/



-- set sql_safe_updates = 0;


/*************************
UPDATE waterpoint
SET subjective_quality = 'unknown',
    subjective_quality_clean = 'unknown'
WHERE subjective_quality IS NOT NULL
  AND subjective_quality_clean IS NOT NULL
  AND TRIM(subjective_quality) = ''
  AND TRIM(subjective_quality_clean) = '';
*********************************/


-- set sql_safe_updates = 1;


/**************************************
ALTER TABLE waterpoint
MODIFY COLUMN subjective_quality VARCHAR(150),
MODIFY COLUMN subjective_quality_clean VARCHAR(100);
***************************************/




/*********************************
SELECT activity_id, COUNT(*) AS cnt
FROM waterpoint
GROUP BY activity_id
ORDER BY cnt DESC;				= there is already row_id which is primary key so activity_id column can be droped
***********************************/



/***************
SELECT activity_id, scheme_id, COUNT(*) AS cnt
FROM waterpoint
GROUP BY activity_id, scheme_id
ORDER BY cnt DESC;	
*******************/



/***************
ALTER TABLE waterpoint
DROP COLUMN activity_id;
*****************/

													-- droped both columns because we already hase row_id and which is primary key
/*************
ALTER TABLE waterpoint
DROP COLUMN scheme_id;
******************/



/*************************
SELECT notes, COUNT(*) AS cnt
FROM waterpoint
GROUP BY notes
ORDER BY cnt DESC;
***********************/


-- in the notes mostly locations are provided and for that we already have its latitude and longatitudes and because of this notes column has been dropped


/***************
ALTER TABLE waterpoint
DROP COLUMN notes;
*****************/


/************
SELECT photo_lnk, COUNT(*) AS cnt
FROM waterpoint
GROUP BY photo_lnk
ORDER BY cnt DESC;
****************/

/****************
ALTER TABLE waterpoint
MODIFY COLUMN photo_lnk VARCHAR(255),
MODIFY COLUMN orig_lnk VARCHAR(255),
MODIFY COLUMN data_lnk VARCHAR(255);
**********************/



-- set sql_safe_updates = 0;



/***************
UPDATE waterpoint
SET photo_lnk = 'Not available'
WHERE photo_lnk IS NOT NULL
  AND TRIM(photo_lnk) = '';
**********************/


/***************
SELECT orig_lnk, COUNT(*) AS cnt
FROM waterpoint
GROUP BY orig_lnk 					= As 703050 rows are empty this column can be droped
ORDER BY cnt DESC;
**********************/




/************
ALTER TABLE waterpoint
DROP COLUMN orig_lnk;
****************/


/**************
SELECT data_lnk, COUNT(*) AS cnt
FROM waterpoint
GROUP BY data_lnk
**********************/

-- set sql_safe_updates = 1;






/****************
SELECT country_name, country_id, clean_country_id, clean_country_name, COUNT(*) AS cnt
FROM waterpoint
GROUP BY country_name, country_id, clean_country_id, clean_country_name
*********************/


/*******************
SELECT clean_adm1, clean_adm2, clean_adm3, clean_adm4, COUNT(*) AS cnt
FROM waterpoint
GROUP BY clean_adm1, clean_adm2, clean_adm3, clean_adm4
*****************************/





/*****************************
SELECT water_source_clean, water_source_category, COUNT(*) AS cnt
FROM waterpoint
GROUP BY water_source_clean, water_source_category;
**************************/



-- set sql_safe_updates = 0;

/*********************
ALTER TABLE waterpoint
DROP COLUMN water_source_category;				= water_source_category column is droped because water_source_clean column is more precise
***********************/


/****************
SELECT water_source_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY water_source_clean;
********************/


/*********************
UPDATE waterpoint
SET water_source = 'unknown'
WHERE water_source IS NOT NULL
  AND TRIM(water_source) = '';				= this was a mistake, wrong column selected but as the empty space of this column was already filled with unknown , no rows were harmed
**********************/	


/****************
SELECT water_source, COUNT(*) AS cnt
FROM waterpoint
GROUP BY water_source;			= checked to confirm
***********************/

/**************
UPDATE waterpoint
SET water_source_clean = 'unknown'
WHERE water_source_clean IS NOT NULL
  AND TRIM(water_source_clean) = '';
************************/



/***************
ALTER TABLE waterpoint
MODIFY COLUMN water_source_clean VARCHAR(100);
***************/


/***********
SELECT water_tech_clean, water_tech_category, COUNT(*) AS cnt
FROM waterpoint
GROUP BY water_tech_clean, water_tech_category;
****************/

/***************************
UPDATE waterpoint
SET water_tech_category = 'Hand Pump'
WHERE water_tech_clean IN ('Hand Pump - Rope', 'Hydram Pump');
************************/

/********************
SELECT water_tech_clean, water_tech_category, COUNT(*) AS cnt
FROM waterpoint
GROUP BY water_tech_clean, water_tech_category;
**************************/


/**************************
ALTER TABLE waterpoint
DROP COLUMN water_tech_clean;				= droped because the column water_tech_category column has similar values
*****************************/


/**************************
ALTER TABLE waterpoint
MODIFY COLUMN water_tech_category VARCHAR(100);
***************************/


/*********************
SELECT water_tech, water_tech_category, COUNT(*) AS cnt
FROM waterpoint
GROUP BY water_tech, water_tech_category;
**********************/


/***********************
SELECT facility_type, COUNT(*) AS cnt			= no null values, no empty spaces
FROM waterpoint
GROUP BY facility_type;
********************/


/************************
SELECT count, COUNT(*) AS cnt 
FROM waterpoint
GROUP BY count;
***********************/


/****************
SELECT converted, COUNT(*) AS cnt
FROM waterpoint
GROUP BY converted;
******************/



/*************
ALTER TABLE waterpoint
DROP COLUMN converted;				= values in the columns were unnecessary values
******************/


/********************
SELECT dataset_title, COUNT(*) AS cnt
FROM waterpoint
GROUP BY dataset_title;
**********************/


/*****************
ALTER TABLE waterpoint
DROP COLUMN public_data_source,
DROP COLUMN created_timestamp,
DROP COLUMN updated,
DROP COLUMN dataset_title;				= dropped this because we have already taken data from water point data, creation timestamp and updated time stamp is not required, and dataset_title is not required from where the data is taken
*********************/


/********************************
ALTER TABLE waterpoint
MODIFY COLUMN water_source_clean VARCHAR(100)
AFTER water_source;
********************************/


/******************************
ALTER TABLE waterpoint
MODIFY COLUMN water_tech_category VARCHAR(100)
AFTER water_tech;
******************************/

/***************
SELECT water_source, water_source_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY water_source, water_source_clean;
********************/

/***************
ALTER TABLE waterpoint
MODIFY COLUMN country_id TEXT
AFTER country_name;
****************/


/***************************
ALTER TABLE waterpoint
MODIFY COLUMN clean_country_id TEXT
AFTER country_id;
***********************/



/******************************
ALTER TABLE waterpoint
MODIFY COLUMN clean_country_name TEXT
AFTER clean_country_id;
********************************/



/*****************
ALTER TABLE waterpoint
MODIFY COLUMN clean_adm1 TEXT
AFTER adm2;
********************/


/*********************
ALTER TABLE waterpoint
MODIFY COLUMN clean_adm2 TEXT
AFTER clean_adm1;
********************/


/*********************************
ALTER TABLE waterpoint
MODIFY COLUMN clean_adm3 TEXT
AFTER clean_adm2;
********************************/


/***********************************
ALTER TABLE waterpoint
MODIFY COLUMN clean_adm4 TEXT
AFTER clean_adm3;
*******************************/




/******************
SELECT water_source, water_source_clean, COUNT(*) AS cnt
FROM waterpoint
GROUP BY water_source, water_source_clean;
********************/


/***************************
SELECT water_tech, water_tech_category, COUNT(*) AS cnt
FROM waterpoint
GROUP BY water_tech, water_tech_category;
****************************/

/*****************
SELECT country_name, country_id, clean_country_id, clean_country_name, COUNT(*) AS cnt
FROM waterpoint
GROUP BY country_name, country_id, clean_country_id, clean_country_name;
*******************/



/**********
SELECT lat_deg, lon_deg, COUNT(*) AS cnt
FROM waterpoint
GROUP BY lat_deg, lon_deg;
**************/


/*********************
SELECT
    COUNT(*) AS total_rows,
    SUM(lat_deg IS NULL) AS lat_null_count,
    SUM(lon_deg IS NULL) AS lon_null_count
FROM waterpoint;
**********************/



/***************************
SELECT
    SUM(TRIM(lat_deg) = '') AS lat_empty_count,
    SUM(TRIM(lon_deg) = '') AS lon_empty_count
FROM waterpoint
WHERE lat_deg IS NOT NULL
   OR lon_deg IS NOT NULL;
**************************/


/****************
SELECT lat_deg, lon_deg, COUNT(*) AS cnt
FROM waterpoint
GROUP BY lat_deg, lon_deg;
*****************/

/*****************
SELECT *
FROM waterpoint
WHERE lat_deg NOT BETWEEN -90 AND 90
   OR lon_deg NOT BETWEEN -180 AND 180
   OR lat_deg IS NULL
   OR lon_deg IS NULL;
**********************/


/*************
SELECT row_id, lat_deg, lon_deg
FROM waterpoint
*************/

/********
SELECT row_id, COUNT(*) AS uniquw_count
FROM waterpoint
group by row_id;
*************/

/*********
SELECT MAX(row_id) AS max_id
FROM waterpoint;
**************/


/*******************
SELECT lat_deg, lon_deg, COUNT(*) AS cnt
FROM waterpoint
GROUP BY lat_deg, lon_deg;
********************/



/******************
SELECT row_id, lat_deg, lon_deg
FROM waterpoint
******************/



/************************
SELECT COUNT(*) FROM waterpoint;
SELECT COUNT(*) FROM exact_location;
****************************/

/********************
SELECT COUNT(*)
FROM waterpoint
JOIN exact_location
ON waterpoint.row_id = exact_location.row_id;
*******************/


/************************
CREATE TABLE waterpoint_exact_location AS
SELECT 
    waterpoint.*, 
    exact_location.lat_deg AS lat_deg_exact,
    exact_location.lon_deg AS lon_deg_exact,
    exact_location.ADMIN,
    exact_location.IOS_A3
FROM waterpoint
INNER JOIN exact_location
ON waterpoint.row_id = exact_location.row_id;
**************************/



/********************
SELECT COUNT(*) FROM waterpoint;
SELECT COUNT(*) FROM exact_location;
SELECT COUNT(*) FROM waterpoint_exact_location;
**********************/


select * from waterpoint 
