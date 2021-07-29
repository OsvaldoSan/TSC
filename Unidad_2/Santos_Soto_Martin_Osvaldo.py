%pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext,SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import os
import subprocess
import pandas as pd

from functools import reduce
from datetime import date
import datetime


filename="hdfs://"+"/user/hive/datos/data.csv"
sparkSession = SparkSession.builder.appName("bigdatita").getOrCreate()
df = sparkSession.read.csv(filename,inferSchema=False,header=True)
df.printSchema()


aux=df.select("year","week","tripduration","from_station_id","to_station_id")
aux=aux.withColumn("ruta" ,F.concat_ws("|","from_station_id","to_station_id") )
aux=aux.drop("from_station_id","to_station_id")

# Catalogo de fechas
alfa=aux.select("year","week")
alfa=alfa.drop_duplicates()
alfa=alfa.withColumn("ancla2",F.concat("year","week"))
alfa=alfa.orderBy("year","week")
alfa=alfa.drop("year","week")
alfa=alfa.withColumn("id_fh",F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))

fh_i=1
fh_f=alfa.count()

aux=aux.withColumn("ancla",F.concat("year","week") ) 
aux=aux.drop("year","week")
aux=aux.join(alfa, aux.ancla == alfa.ancla2, "inner")
aux=aux.drop("ancla2")
aux=aux.withColumn("duracion",F.col("tripduration").cast('double') )
aux=aux.drop("tripduration")
aux=aux.withColumn("viaje",F.lit(1))


vobs=12
vdes=1
step=3
anclai,anclaf=fh_i+vobs-1,fh_f-vdes # Cada ancal es una semana
anclai,anclaf


df=aux.drop("ancla")
um=['ruta','id_fh']
df.show()


def ing(df,k,ancla):
    u=df.filter( ( df['id_fh']>= (ancla-k+1) ) & (df['id_fh']<=ancla) ).orderBy("id_fh")
    expr = [F.sum(F.col('viaje')).alias(f'x_num_tot_viajes_{k}')]
    expr.append(F.mean(F.col('duracion')).alias(f'x_duracion_prom_viaje_{k}'))
    u = u.groupBy('ruta').agg(*expr).withColumn('id_fh',F.lit(ancla))
    return u
    

def ing_tgt(df,ancla):
    u=df.filter( df['id_fh']== ancla + 1 ).orderBy("id_fh")
    expr = [F.sum(F.col('viaje')).alias('prediccion')]
    
    u = u.groupBy('ruta').agg(*expr).withColumn('id_fh',F.lit(ancla))
    return u

# Adaptación a programación iterativa
u1=reduce(lambda x,y:x.join(y,um,'outer'),map(lambda k:ing(df,k,anclai),range(step,vobs+step,step)))
u1.show()


for i in range(anclai+1,anclaf+1):
    u2=reduce(lambda x,y:x.join(y,um,'outer'),map(lambda k:ing(df,k,i),range(step,vobs+step,step)))
    u2.show(1)
    u1=u1.union(u2)
    u1.show(1)
    
    
v=reduce(lambda x,y:x.union(y), map(lambda ancla:ing_tgt(df,ancla),range(anclai,anclaf+1)) )

u=u1.join(v,um,'inner')

r=u.orderBy("id_fh")

tad=r.join(alfa,'id_fh',"inner")
tad=tad.withColumnRenamed('ancla2','ancla')
tad=tad.select('ruta','ancla','x_num_tot_viajes_3','x_num_tot_viajes_6','x_num_tot_viajes_9',
           'x_num_tot_viajes_12','x_duracion_prom_viaje_3','x_duracion_prom_viaje_6',
           'x_duracion_prom_viaje_9','x_duracion_prom_viaje_12','prediccion')
tad.show(1)

tad.coalesce(1).write.option("header", "true").csv('/tad')

