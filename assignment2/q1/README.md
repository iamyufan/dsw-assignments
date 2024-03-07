# Question 1: Extract, Transform Load (ETL)

## Step 1: Load the data

```python
def create_dataframe(filepath, format, spark):
    """
    Create a spark df given a filepath and format.

    :param filepath: <str>, the filepath
    :param format: <str>, the file format (e.g. "csv" or "json")
    :param spark: <str> the spark session

    :return: the spark df uploaded
    """

    #add your code here
    spark_df = spark.read.format(format).load(filepath)

    return spark_df
```

## Step 2: Make test example

**Analyze the columns**

BRFSS data has 4 columns:

1. `SEX`: Respondents Sex
   1. 1: Male
   2. 2: Female
   3. 9: Refused
2. `_LLCPWT`: Final weight: Land-line and cell-phone data
3. `_AGEG5YR`: Reported age in five-year age categories calculated variable
   1. 1: 18-24
   2. 2: 25-29
   3. 3: 30-34
   4. 4: 35-39
   5. 5: 40-44
   6. 6: 45-49
   7. 7: 50-54
   8. 8: 55-59
   9. 9: 60-64
   10. 10: 65-69
   11. 11: 70-74
   12. 12: 75-79
   13. 13: 80-99
   14. 14: Don’t know/Refused/Missing
4. `_IMPRACE`: Imputed race/ethnicity value
   1. 1: White, Non-Hispanic
   2. 2: Black, Non-Hispanic
   3. 3: Asian, Non-Hispanic
   4. 4: American Indian/Alaskan Native, Non-Hispanic
   5. 5: Hispanic
   6. 6: Other race, Non-Hispanic

NHIS data has 5 columns:

1. `SEX`: male or female
   1. 1: Male
   2. 2: Female
2. `MRACBPI2`: race
   1. 01: White
   2. 02: Black/African American
   3. 03: Indian (American) (includes Eskimo, Aleut)
   4. 06: Chinese
   5. 07: Filipino
   6. 12: Asian Indian
   7. 16: Other race
   8. 17: Multiple race, no primary race selected
3. `HISPAN_I`: ethnicity
   1. 00: Multiple Hispanic
   2. 01: Puerto Rico
   3. 02: Mexican
   4. 03: Mexican-American
   5. 04: Cuban/Cuban American
   6. 05: Dominican (Republic)
   7. 06: Central or South American
   8. 07: Other Latin American, type not specified
   9. 08: Other Spanish
   10. 09: Hispanic/Latino/Spanish, non-specific type
   11. 10: Hispanic/Latino/Spanish, type refused
   12. 11: Hispanic/Latino/Spanish, type not ascertained
   13. 12: Not Hispanic/Spanish origin
4. `AGE_P`: age
   1. 00: Under 1 year
   2. 01-84: 1-84 years
   3. 85: 85 years and over
5. `DIBEV1`: Have you EVER been told by a doctor or health professional that you have diabetes or sugar diabetes?
   1. 1: Yes
   2. 2: No
   3. 3: Borderline or prediabetes
   4. 7: Refused
   5. 8: Not ascertained
   6. 9: Don't know

**Test data**

`test_brfss.json`:

```
{"SEX":2.0,"_LLCPWT":1,"_AGEG5YR":11.0,"_IMPRACE":1.0}  # Female, 70-74, White
{"SEX":1.0,"_LLCPWT":2,"_AGEG5YR":6.0,"_IMPRACE":1.0}   # Male, 45-49, White
{"SEX":2.0,"_LLCPWT":2,"_AGEG5YR":10.0,"_IMPRACE":5.0}  # Female, 65-69, Hispanic
{"SEX":2.0,"_LLCPWT":1,"_AGEG5YR":1.0,"_IMPRACE":1.0}   # Female, 18-24, White
{"SEX":2.0,"_LLCPWT":1,"_AGEG5YR":11.0,"_IMPRACE":1.0}  # Female, 70-74, White
```

`test_nhis.csv`:

```
SEX,MRACBPI2,HISPAN_I,AGE_P,DIBEV1
,1,12,65,2
2,1,12,19,2 # Female, White, Not Hispanic, 18-24, 2
1,1,12,45,1 # Male, White, Not Hispanic, 45-54, 1
2,1,0,67,1  # Female, White, Multiple Hispanic, 65-74, 1
1,1,12,,2
```

`joined_test.csv`:

```
SEX,_LLCPWT,_AGEG5YR,_IMPRACE,DIBEV1
1.0,2,6.0,1.0,1
2.0,2,10.0,5.0,1
2.0,1,1.0,1.0,2
```

## Step 3: Map data

## Step 4: Join data

```bash
python3 p1.py data/test_nhis.csv data/test_brfss.json
```

Result on the test data:

```
+---+--------+--------+-------+------+
|SEX|_AGEG5YR|_IMPRACE|_LLCPWT|DIBEV1|
+---+--------+--------+-------+------+
|1.0|     6.0|     1.0|      2|   1.0|
|2.0|    10.0|     5.0|      2|   1.0|
|2.0|     1.0|     1.0|      1|   2.0|
+---+--------+--------+-------+------+
```

```bash
python3 p1.py data/nhis_input.csv data/brfss_input.json
```

## Step 5: Report prevalence

Prevalence statistics for `_IMPRACE`:

```bash
+--------+---------+------------+------------------+
|_IMPRACE|total    |diabetes_yes|prevalence        |
+--------+---------+------------+------------------+
|1.0     |261475361|29450159    |11.263072316783225|
|2.0     |4263219  |683486      |16.032157860058327|
|3.0     |380268   |22140       |5.822209599545584 |
|4.0     |97158    |22928       |23.598674324296507|
|5.0     |5320518  |474787      |8.923698782712512 |
|6.0     |309723   |28446       |9.184335680592012 |
+--------+---------+------------+------------------+
```

- 1: White, Non-Hispanic
- 2: Black, Non-Hispanic
- 3: Asian, Non-Hispanic
- 4: American Indian/Alaskan Native, Non-Hispanic
- 5: Hispanic
- 6: Other race, Non-Hispanic

True prevalence statistics [[ref: CDC](https://www.cdc.gov/diabetes/data/statistics-report/index.html)]:

- White, Non-Hispanic: 8.5
- Black, Non-Hispanic: 12.5
- Asian, Non-Hispanic: 9.2
- American Indian/Alaskan Native, Non-Hispanic: 16.0
- Hispanic: 10.3

Prevalence statistics for `SEX`:

```bash
+---+---------+------------+------------------+
|SEX|total    |diabetes_yes|prevalence        |
+---+---------+------------+------------------+
|1.0|108477351|14266074    |13.151200567204116|
|2.0|163368896|16415872    |10.04834604501459 |
+---+---------+------------+------------------+
```

True prevalence statistics [[ref: CDC](https://www.cdc.gov/diabetes/data/statistics-report/index.html)]:

- Men: 12.6
- Women: 10.2

Prevalence statistics for `_AGEG5YR`:

```bash
+--------+--------+------------+------------------+
|_AGEG5YR|total   |diabetes_yes|prevalence        |
+--------+--------+------------+------------------+
|1.0     |12965168|126534      |0.9759534161069104|
|2.0     |10061594|147302      |1.4640026222485225|
|3.0     |12086977|324417      |2.6840209921802614|
|4.0     |12314645|431783      |3.5062561689760443|
|5.0     |10875373|471310      |4.333736415293526 |
|6.0     |15460714|1207538     |7.810363738699261 |
|7.0     |22690546|2096156     |9.23801481022096  |
|8.0     |32106924|3853245     |12.00128981524359 |
|9.0     |39079332|5255536     |13.448377264995216|
|10.0    |38454540|5912198     |15.374512346266528|
|11.0    |26970563|5007216     |18.5654856370629  |
|12.0    |13317955|2381834     |17.88438239955008 |
|13.0    |25461916|3466877     |13.615931338395743|
+--------+--------+------------+------------------+
```

After further calculation:

```json
{
  "18-44": {
    "total": 58303757,
    "diabetes_yes": 1501346,
    "prevalence": 2.5750416049518043
  },
  "45-64": {
    "total": 109337516,
    "diabetes_yes": 12412475,
    "prevalence": 11.3524391755891
  },
  "65+": {
    "total": 104204974,
    "diabetes_yes": 16768125,
    "prevalence": 16.091482350928853
  }
}
```

True prevalence statistics [[ref: CDC](https://www.cdc.gov/diabetes/data/statistics-report/index.html)]:

- 18-24: 3.0
- 45–64: 14.5
- \>=65: 24.4