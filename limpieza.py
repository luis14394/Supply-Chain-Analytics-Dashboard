import pandas as pd
import numpy as np

# CARGA DE DATOS
path = r'C:\two\Data'
df_main = pd.read_csv(path + r'\DataCoSupplyChainDataset.csv', encoding='latin-1', low_memory=False)
df_desc = pd.read_csv(path + r'\DescriptionDataCoSupplyChain.csv', encoding='latin-1')
df_logs = pd.read_csv(path + r'\tokenized_access_logs.csv', encoding='latin-1')

print("="*60)
print("ETL SUPPLY CHAIN - INICIANDO PROCESO")
print("="*60)

# NORMALIZACIÓN DE COLUMNAS
df_main.columns = df_main.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
df_desc.columns = df_desc.columns.str.strip().str.lower()
df_logs.columns = df_logs.columns.str.strip().str.lower()

# LIMPIEZA LOGS
df_logs = df_logs.drop_duplicates()
df_logs['ip'] = df_logs.get('ip', pd.Series()).fillna('0.0.0.0')

# LIMPIEZA DATASET PRINCIPAL
null_threshold = len(df_main) * 0.85
df_main = df_main.dropna(thresh=null_threshold, axis=1)

# CORRECCIÓN VALORES NEGATIVOS
columnas_numericas = ['sales', 'benefit_per_order', 'sales_per_customer', 'order_item_product_price']
df_main[columnas_numericas] = df_main[columnas_numericas].clip(lower=0)

# IMPUTACIÓN DE NULOS
df_main['order_item_discount'] = df_main['order_item_discount'].fillna(0)
df_main['order_item_discount_rate'] = df_main.get('order_item_discount_rate', pd.Series()).fillna(0)

# NORMALIZACIÓN DE NOMBRES DE PRODUCTOS
df_main['product_name'] = df_main['product_name'].str.upper().str.strip()

print("Datos cargados y limpios - Creando KPIs...")

# FEATURES LOGÍSTICAS
df_main['lead_time_gap'] = df_main['days_for_shipping_real'] - df_main['days_for_shipment_scheduled']

# CLASIFICACIÓN DESEMPEÑO ENTREGA
delivery_cond = [df_main['lead_time_gap'] > 2, 
                 df_main['lead_time_gap'].between(-1, 2), 
                 df_main['lead_time_gap'] < -1]
delivery_labels = ['Retraso_Critico', 'On_Time', 'Anticipado']
df_main['delivery_performance'] = np.select(delivery_cond, delivery_labels, default='Sin_Info')

# EFICIENCIA ENTREGA
df_main['delivery_efficiency'] = np.where(df_main['days_for_shipment_scheduled'] > 0,
                                           100 * (1 - df_main['lead_time_gap'].abs() / df_main['days_for_shipment_scheduled']),
                                           0)
df_main['delivery_efficiency'] = df_main['delivery_efficiency'].clip(0, 100)

# MARGEN DE GANANCIA
df_main['profit_margin_pct'] = np.where(df_main['sales'] > 0,
                                         100 * (df_main['benefit_per_order'] / df_main['sales']),
                                         0)

# VALOR PROMEDIO PEDIDO
df_main['avg_order_value'] = df_main['sales'] / df_main['order_item_quantity'].replace(0, 1)

# CLASIFICACIÓN RENTABILIDAD
profit_cond = [df_main['order_item_profit_ratio'] <= 0,
               df_main['order_item_profit_ratio'] <= 0.15,
               df_main['order_item_profit_ratio'] <= 0.35,
               df_main['order_item_profit_ratio'] > 0.35]
profit_labels = ['Sin_Ganancia', 'Baja', 'Media', 'Alta']
df_main['rentabilidad'] = np.select(profit_cond, profit_labels, default='ND')

# ANÁLISIS DE RIESGO
risk_cond = [(df_main['order_status'] == 'SUSPECTED_FRAUD') & (df_main['sales'] > 1000),
             (df_main['order_status'] == 'SUSPECTED_FRAUD'),
             (df_main['order_status'] == 'CANCELED') & (df_main['sales'] > 500),
             df_main['late_delivery_risk'] == 1]
risk_labels = ['Fraude_Alto', 'Fraude_Bajo', 'Cancelacion_Sospechosa', 'Riesgo_Entrega']
df_main['categoria_riesgo'] = np.select(risk_cond, risk_labels, default='Normal')

# INDICADOR BINARIO RIESGO
df_main['es_alto_riesgo'] = df_main['categoria_riesgo'].isin(['Fraude_Alto', 'Fraude_Bajo']).astype(int)

print("KPIs logísticos y financieros creados - Segmentación...")

# SEGMENTACIÓN CLIENTES RFM
df_main['cliente_frecuencia'] = df_main.groupby('customer_id')['order_id'].transform('count')
df_main['cliente_valor_total'] = df_main.groupby('customer_id')['sales'].transform('sum')

# CLASIFICACIÓN CLIENTES
customer_cond = [(df_main['cliente_frecuencia'] >= 10) & (df_main['cliente_valor_total'] >= 5000),
                 (df_main['cliente_frecuencia'] >= 5) & (df_main['cliente_valor_total'] >= 1000),
                 df_main['cliente_frecuencia'] >= 2]
customer_labels = ['VIP', 'Frecuente', 'Ocasional']
df_main['segmento_cliente'] = np.select(customer_cond, customer_labels, default='Nuevo')

# CLASIFICACIÓN ABC INVENTARIO
ventas_por_producto = df_main.groupby('product_name')['sales'].transform('sum')
percentil_95 = df_main.groupby('product_name')['sales'].sum().quantile(0.95)
percentil_80 = df_main.groupby('product_name')['sales'].sum().quantile(0.80)

abc_cond = [ventas_por_producto >= percentil_95,
            ventas_por_producto >= percentil_80,
            ventas_por_producto < percentil_80]
abc_labels = ['A_Premium', 'B_Medio', 'C_Basico']
df_main['categoria_abc'] = np.select(abc_cond, abc_labels, default='Sin_Clase')

# MÉTRICAS GEOGRÁFICAS
df_main['ventas_pais'] = df_main.groupby('order_country')['sales'].transform('sum')
df_main['participacion_mercado'] = (df_main['ventas_pais'] / df_main['sales'].sum()) * 100

print("Segmentación completada - Preparando exportación...")

# ELIMINAR COLUMNAS SENSIBLES
columnas_eliminar = ['customer_password', 'customer_fname', 'customer_lname', 
                     'customer_street', 'customer_email']
df_main = df_main.drop(columns=[col for col in columnas_eliminar if col in df_main.columns])

# EXPORTACIÓN FINAL
archivo_salida = path + r'\SupplyChain_Master_Dashboard.csv'
df_main.to_csv(archivo_salida, index=False, encoding='utf-8-sig')

print("="*60)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("="*60)
print(f"Archivo: SupplyChain_Master_Dashboard.csv")
print(f"Dimensiones: {df_main.shape[0]:,} filas x {df_main.shape[1]} columnas")
print(f"Clientes únicos: {df_main['customer_id'].nunique():,}")
print(f"Productos únicos: {df_main['product_name'].nunique():,}")
print(f"Órdenes únicas: {df_main['order_id'].nunique():,}")
print(f"Ventas totales: ${df_main['sales'].sum():,.2f}")
print("="*60)
print("KPIs PARA POWER BI:")
print("- lead_time_gap, delivery_performance, delivery_efficiency")
print("- profit_margin_pct, avg_order_value, rentabilidad")
print("- categoria_riesgo, es_alto_riesgo, categoria_abc")
print("- segmento_cliente, cliente_frecuencia, cliente_valor_total")
print("- participacion_mercado, ventas_pais")
print("="*60)