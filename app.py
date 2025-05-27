from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.optimize import linprog
import os
import plotly.graph_objs as go
import plotly.io as pio
from collections import defaultdict
from plotly.subplots import make_subplots
import sys


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static')

def parse_func_objetivo(texto, num_vars):
    texto = texto.replace('-', '+-')
    partes = texto.split('+')
    coef = [0] * num_vars
    var_letras = ['x', 'y', 'z']
    for parte in partes:
        parte = parte.strip()
        for i, var in enumerate(var_letras[:num_vars]):
            if var in parte:
                num = parte.replace(var, '') or '1'
                try:
                    coef[i] = float(num)
                except:
                    raise ValueError(f"Coeficiente inválido para {var}: '{num}'")
    return coef

def parse_restriccion(texto, num_vars):
    texto = texto.replace('-', '+-')
    if '<=' in texto:
        izq, der = texto.split('<=', 1)
        tipo = '<='
    elif '>=' in texto:
        izq, der = texto.split('>=', 1)
        tipo = '>='
    elif '=' in texto:
        izq, der = texto.split('=', 1)
        tipo = '='
    else:
        raise ValueError("Restricción debe contener <=, >= o =")

    coef = [0] * num_vars
    partes = izq.split('+')
    var_letras = ['x', 'y', 'z']
    for parte in partes:
        parte = parte.strip()
        for i, var in enumerate(var_letras[:num_vars]):
            if var in parte:
                num = parte.replace(var, '') or '1'
                try:
                    coef[i] = float(num)
                except:
                    raise ValueError(f"Coeficiente inválido en restricción para {var}: '{num}'")
    val = float(der.strip())
    return coef, val, tipo

def calcular_vertices_2d(A_ub, b_ub):
    """Calcula todos los vértices de la región factible con precisión numérica robusta"""
    vertices = []
    n = len(A_ub)
    tol = 1e-10  # Tolerancia para comparaciones numéricas
    
    # 1. Intersecciones con los ejes (puntos de frontera)
    for i in range(n):
        # Intersección con eje x (y=0)
        if abs(A_ub[i][0]) > tol:
            x = b_ub[i] / A_ub[i][0]
            y = 0.0
            if x >= -tol and all(a[0]*x + a[1]*y <= b + tol for a, b in zip(A_ub, b_ub)):
                vertices.append((x, y))
        
        # Intersección con eje y (x=0)
        if abs(A_ub[i][1]) > tol:
            x = 0.0
            y = b_ub[i] / A_ub[i][1]
            if y >= -tol and all(a[0]*x + a[1]*y <= b + tol for a, b in zip(A_ub, b_ub)):
                vertices.append((x, y))
    
    # 2. Intersecciones entre pares de restricciones
    for i in range(n):
        for j in range(i+1, n):
            a1, b1, c1 = A_ub[i][0], A_ub[i][1], b_ub[i]
            a2, b2, c2 = A_ub[j][0], A_ub[j][1], b_ub[j]
            
            # Determinante para resolver el sistema lineal
            det = a1*b2 - a2*b1
            if abs(det) < tol:  # Restricciones paralelas o coincidentes
                continue
                
            x = (b2*c1 - b1*c2) / det
            y = (a1*c2 - a2*c1) / det
            
            # Verificación de factibilidad con tolerancia numérica
            if (x >= -tol and y >= -tol and 
                all(a[0]*x + a[1]*y <= b + tol for a, b in zip(A_ub, b_ub))):
                vertices.append((x, y))
    
    # 3. Procesamiento post-cálculo de vértices
    if vertices:
        # Eliminar duplicados usando numpy con precisión controlada
        vertices = np.array(vertices)
        vertices = np.unique(np.round(vertices, decimals=8), axis=0)
        
        # Ordenar vértices en sentido horario para visualización
        if len(vertices) > 2:
            centroide = np.mean(vertices, axis=0)
            angulos = np.arctan2(vertices[:,1]-centroide[1], vertices[:,0]-centroide[0])
            vertices = vertices[np.argsort(-angulos)]
    
    return vertices

def graficar_2d_plotly(A, b, vertices, res, maximizar, nombres_variables=None):
    """Genera un gráfico interactivo profesional con visualización mejorada"""
    # Configuración de estilo profesional mejorada
    colors = {
        'fondo': 'rgb(245, 245, 245)',
        'ejes': 'rgb(80, 80, 80)',
        'grid': 'rgba(180, 180, 180, 0.3)',
        'restricciones': ['#3366CC', '#DC3912', '#FF9900', '#109618', '#990099'],
        'factible': 'rgba(91, 155, 213, 0.25)',
        'optimo': '#FF33AA',
        'texto': 'rgb(40, 40, 40)',
        'vertices': '#2A52BE'
    }
    
    # Crear figura con configuración profesional
    fig = go.Figure()
    
    # 1. Configuración de rangos dinámicos
    x_vals = []
    y_vals = []
    
    for i in range(len(A)):
        if abs(A[i][0]) > 1e-10:
            x_vals.append(b[i]/A[i][0])
        if abs(A[i][1]) > 1e-10:
            y_vals.append(b[i]/A[i][1])
    
    if res.success:
        x_vals.extend([res.x[0]*1.5])
        y_vals.extend([res.x[1]*1.5])
    
    safe_max = lambda vals: max(vals) if vals else 10
    x_range = [0, safe_max(x_vals)*1.15]
    y_range = [0, safe_max(y_vals)*1.15]
    
    # 2. Curvas de nivel (solo si hay solución)
    if res.success:
        x_grid = np.linspace(x_range[0], x_range[1], 100)
        y_grid = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = res.x[0]*X + res.x[1]*Y
        
        fig.add_trace(go.Contour(
            x=x_grid, y=y_grid, z=Z,
            contours=dict(
                start=0,
                end=np.max(Z)*1.1,
                size=np.max(Z)/10,
                coloring='lines',
                showlabels=True
            ),
            line=dict(width=1.5, color='rgba(100,100,100,0.4)'),
            opacity=0.6,
            showscale=False,
            name='Curvas de nivel',
            hoverinfo='skip'
        ))
    
    # 3. Restricciones con etiquetas inteligentes
    for i, (coef, val) in enumerate(zip(A, b)):
        a, c = coef[0], coef[1]
        color = colors['restricciones'][i % len(colors['restricciones'])]
        
        # Posicionamiento inteligente de etiquetas
        label_x = x_range[1] * 0.65
        label_y = (val - a * label_x) / c if abs(c) > 1e-10 else y_range[1] * 0.85
        
        if abs(c) > 1e-10:  # Restricción no vertical
            y_vals = (val - a * x_grid) / c
            y_vals = np.clip(y_vals, y_range[0], y_range[1])
            
            fig.add_trace(go.Scatter(
                x=x_grid, y=y_vals,
                mode='lines',
                line=dict(color=color, width=2.5, dash='dash'),
                name=f'Restricción {i+1}',
                hoverinfo='text',
                hovertext=f'{a:.2f}x₁ + {c:.2f}x₂ ≤ {val:.2f}',
                opacity=0.9
            ))
        else:  # Restricción vertical
            x_line = val / a
            fig.add_trace(go.Scatter(
                x=[x_line, x_line], y=y_range,
                mode='lines',
                line=dict(color=color, width=2.5, dash='dash'),
                name=f'x₁ ≤ {x_line:.2f}',
                hoverinfo='text',
                hovertext=f'x₁ ≤ {x_line:.2f}'
            ))
        
        # Anotaciones mejoradas para restricciones
        fig.add_annotation(
            x=label_x,
            y=label_y,
            text=f'{a:.2f}x₁ + {c:.2f}x₂ ≤ {val:.2f}',
            showarrow=False,
            font=dict(size=11, color=color),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor=color,
            borderwidth=1.5,
            borderpad=3
        )
    
    # 4. Región factible (si existe)
    if len(vertices) > 2:
        fig.add_trace(go.Scatter(
            x=np.append(vertices[:, 0], vertices[0, 0]),
            y=np.append(vertices[:, 1], vertices[0, 1]),
            fill='toself',
            fillcolor=colors['factible'],
            line=dict(color='rgb(65,105,225)', width=3),
            mode='lines',
            name='Región factible',
            hoverinfo='text',
            hovertext='Área de soluciones factibles',
            opacity=0.7
        ))
    
    # 5. Vértices con información detallada
    for i, vertex in enumerate(vertices):
        fig.add_trace(go.Scatter(
            x=[vertex[0]],
            y=[vertex[1]],
            mode='markers',
            marker=dict(
                size=10,
                color=colors['vertices'],
                line=dict(width=1.5, color='white')
            ),
            name=f'Vértice {i+1}',
            hoverinfo='text',
            hovertext=f'Vértice {i+1}<br>x₁ = {vertex[0]:.4f}<br>x₂ = {vertex[1]:.4f}',
            showlegend=False
        ))
    
    # 6. Solución óptima con elementos destacados
    if res.success:
        # Punto óptimo
        fig.add_trace(go.Scatter(
            x=[res.x[0]],
            y=[res.x[1]],
            mode='markers+text',
            marker=dict(
                size=18,
                color=colors['optimo'],
                symbol='star-diamond',
                line=dict(width=2, color='black')
            ),
            name='Óptimo',
            text=f"Óptimo: ({res.x[0]:.4f}, {res.x[1]:.4f})",
            textposition="top right",
            hoverinfo='text',
            hovertext=f'''Solución óptima<br>
                        {'Max' if maximizar else 'Min'}: {-res.fun if maximizar else res.fun:.4f}<br>
                        x₁ = {res.x[0]:.4f}<br>
                        x₂ = {res.x[1]:.4f}'''
        ))
        
        # Líneas de referencia al óptimo
        fig.add_shape(type="line",
            x0=res.x[0], y0=0, x1=res.x[0], y1=res.x[1],
            line=dict(color=colors['optimo'], width=2, dash="dot"),
            opacity=0.7
        )
        fig.add_shape(type="line",
            x0=0, y0=res.x[1], x1=res.x[0], y1=res.x[1],
            line=dict(color=colors['optimo'], width=2, dash="dot"),
            opacity=0.7
        )
    
    # 7. Configuración final del layout
    fig.update_layout(
        title={
            'text': "<b>Análisis Gráfico de Programación Lineal</b>",
            'y':0.96,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color=colors['texto'], family="Arial")
        },
        paper_bgcolor=colors['fondo'],
        plot_bgcolor=colors['fondo'],
        xaxis=dict(
            title=dict(
        text='<b>Variable x₁</b>',
        font=dict(size=16, color=colors['texto'], family="Arial")
        ),
        range=x_range,
        showgrid=True,
        gridcolor=colors['grid'],
        zeroline=True,
        zerolinecolor=colors['ejes'],
        linecolor=colors['ejes'],
        mirror=True,
        ticks='outside',
        tickfont=dict(size=12, color=colors['texto'], family="Arial")
        ),
        yaxis=dict(
            title=dict(
        text='<b>Variable x₂</b>',
        font=dict(size=16, color=colors['texto'], family="Arial")
        ),
        range=y_range,
        showgrid=True,
        gridcolor=colors['grid'],
        zeroline=True,
        zerolinecolor=colors['ejes'],
        linecolor=colors['ejes'],
        mirror=True,
        ticks='outside',
        tickfont=dict(size=12, color=colors['texto'], family="Arial")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=12, color=colors['texto'], family="Arial")
        ),
        margin=dict(l=80, r=80, t=100, b=80),
        width=1100,
        height=750,
        hoverlabel=dict(
            bgcolor='white',
            font_size=13,
            font_family="Arial"
        )
    )
    
    # 8. Cuadro de información de resultados
    if res.success:
        resultado_texto = f"""
        <b>RESULTADOS</b><br>
        {'Maximizando' if maximizar else 'Minimizando'}: <b>{-res.fun if maximizar else res.fun:.4f}</b><br>
        x₁ = {res.x[0]:.4f}<br>
        x₂ = {res.x[1]:.4f}<br>
        """
        
        fig.add_annotation(
            x=0.03,
            y=0.97,
            xref='paper',
            yref='paper',
            text=resultado_texto,
            showarrow=False,
            align='left',
            bordercolor=colors['ejes'],
            borderwidth=1.5,
            borderpad=5,
            bgcolor='rgba(255,255,255,0.9)',
            font=dict(size=13, color=colors['texto'], family="Arial")
        )
    
    # Guardar como JSON también
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'grafico2d.json')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        pio.write_json(fig, file=json_path, pretty=True)
        return 'grafico2d_interactivo.html', 'grafico2d.json'



class SimplexSolver:
    def __init__(self, c, A, b, maximizar=True):
        self.maximizar = maximizar
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)
        self.steps = []
        self.solution = None
        self.optimal = False
        
    def to_standard_form(self):
        """Convierte el problema a forma estándar para Simplex"""
        m, n = self.A.shape
        
        # Convertir a maximización si es minimización
        if not self.maximizar:
            self.c = -self.c
            
        # Agregar variables de holgura
        slack_vars = np.eye(m)
        self.A = np.hstack([self.A, slack_vars])
        self.c = np.hstack([self.c, np.zeros(m)])
        
        # Identificadores de variables
        self.var_names = [f'x{i+1}' for i in range(n)] + [f's{i+1}' for i in range(m)]
        self.basic_vars = [f's{i+1}' for i in range(m)]
        
    def create_initial_table(self):
        """Crea la tabla Simplex inicial"""
        m, n = self.A.shape
        self.table = np.zeros((m+1, n+1))
        
        # Llenar restricciones
        self.table[:-1, :-1] = self.A
        self.table[:-1, -1] = self.b
        
        # Llenar función objetivo
        self.table[-1, :-1] = -self.c
        self.table[-1, -1] = 0  # Valor inicial Z
        
        self.current_step = {
            'table': self.table.copy(),
            'basic_vars': self.basic_vars.copy(),
            'entering': None,
            'leaving': None,
            'pivot': None,
            'message': "Tabla inicial"
        }
        self.steps.append(self.current_step)
        
    def is_optimal(self):
        """Verifica si la solución es óptima"""
        return all(x >= 0 for x in self.table[-1, :-1])
    
    def find_pivot(self):
        """Encuentra el elemento pivote"""
        # Variable entrante (más negativo en la fila Z)
        entering_idx = np.argmin(self.table[-1, :-1])
        self.entering_var = self.var_names[entering_idx]
        
        # Cocientes para variable saliente
        ratios = []
        for i in range(len(self.table)-1):
            if self.table[i, entering_idx] > 0:
                ratios.append(self.table[i, -1]/self.table[i, entering_idx])
            else:
                ratios.append(np.inf)
                
        if all(r == np.inf for r in ratios):
            raise ValueError("Problema no acotado")
            
        leaving_idx = np.argmin(ratios)
        self.leaving_var = self.basic_vars[leaving_idx]
        
        return entering_idx, leaving_idx
    
    def pivot(self, entering_idx, leaving_idx):
        """Realiza el pivoteo"""
        pivot_row = leaving_idx
        pivot_col = entering_idx
        pivot_val = self.table[pivot_row, pivot_col]
        
        # Normalizar la fila pivote
        self.table[pivot_row, :] /= pivot_val
        
        # Actualizar otras filas
        for i in range(len(self.table)):
            if i != pivot_row and self.table[i, pivot_col] != 0:
                multiplier = self.table[i, pivot_col]
                self.table[i, :] -= multiplier * self.table[pivot_row, :]
        
        # Actualizar variables básicas
        self.basic_vars[pivot_row] = self.var_names[pivot_col]
        
    def solve_step_by_step(self):
        """Resuelve el problema paso a paso"""
        self.to_standard_form()
        self.create_initial_table()
        
        while not self.is_optimal():
            try:
                entering_idx, leaving_idx = self.find_pivot()
                
                self.current_step = {
                    'table': self.table.copy(),
                    'basic_vars': self.basic_vars.copy(),
                    'entering': self.var_names[entering_idx],
                    'leaving': self.basic_vars[leaving_idx],
                    'pivot': (leaving_idx, entering_idx),
                    'message': f"Iteración {len(self.steps)}: {self.var_names[entering_idx]} entra, {self.basic_vars[leaving_idx]} sale"
                }
                self.steps.append(self.current_step)
                
                self.pivot(entering_idx, leaving_idx)
                
            except ValueError as e:
                self.current_step['message'] = str(e)
                self.steps.append(self.current_step)
                break
                
        # Solución final
        solution = defaultdict(float)
        for i, var in enumerate(self.basic_vars):
            solution[var] = self.table[i, -1]
            
        solution['Z'] = self.table[-1, -1]
        
        if not self.maximizar:
            solution['Z'] = -solution['Z']
            
        self.solution = solution
        self.optimal = self.is_optimal()
        
        self.current_step = {
            'table': self.table.copy(),
            'basic_vars': self.basic_vars.copy(),
            'entering': None,
            'leaving': None,
            'pivot': None,
            'message': "Solución final" if self.optimal else "Problema no acotado o sin solución",
            'solution': solution
        }
        self.steps.append(self.current_step)
        
        return self.steps, self.solution


    
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/simplex', methods=['POST'])
def simplex():
    try:
        num_vars = int(request.form.get('variables'))
        if num_vars != 2:
            return "Solo soporta 2 variables por ahora.", 400

        objetivo = request.form.get('objetivo')
        if not objetivo:
            return "Función objetivo requerida.", 400

        maximizar = (request.form.get('tipo_objetivo', 'max') == 'max')

        restricciones_texto = request.form.get('restricciones', '')
        restricciones = [line.strip() for line in restricciones_texto.strip().split('\n') if line.strip()]
        if len(restricciones) == 0:
            return "Debes ingresar al menos una restricción.", 400

        f_obj = parse_func_objetivo(objetivo, num_vars)

        A_ub, b_ub = [], []
        for r in restricciones:
            coef, val, tipo = parse_restriccion(r, num_vars)
            if tipo == '<=':
                A_ub.append(coef)
                b_ub.append(val)
            elif tipo == '>=':
                A_ub.append([-c for c in coef])
                b_ub.append(-val)
            elif tipo == '=':
                A_ub.append(coef)
                b_ub.append(val)
                A_ub.append([-c for c in coef])
                b_ub.append(-val)

        if len(A_ub) == 0:
            return "No hay restricciones válidas para resolver.", 400

        # Resolver con Simplex paso a paso
        solver = SimplexSolver(f_obj, A_ub, b_ub, maximizar)
        steps, solution = solver.solve_step_by_step()
        
        # Preparar datos para la vista
        simplex_steps = []
        for i, step in enumerate(steps):
            simplex_steps.append({
                'iteration': i,
                'table': step['table'].tolist(),
                'basic_vars': step['basic_vars'],
                'entering': step['entering'],
                'leaving': step['leaving'],
                'pivot': step['pivot'],
                'message': step['message'],
                'var_names': solver.var_names if hasattr(solver, 'var_names') else []
            })
        
        # También resolver con linprog para comparación
        c = [-c for c in f_obj] if maximizar else f_obj
        res = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), method='highs')
        
        resultado = {
            'simplex_steps': simplex_steps,
            'simplex_solution': solution,
            'linprog_solution': {
                'optimal_value': -res.fun if maximizar else res.fun,
                'variables': res.x.tolist(),
                'success': res.success
            },
            'maximizar': maximizar
        }
        
        return render_template('simplex_result.html', resultado=resultado)

    except Exception as e:
        return f"Error procesando la solicitud: {e}", 500
    
    
@app.route('/resolver', methods=['POST'])
def resolver():
    try:
        num_vars = int(request.form.get('variables'))
        if num_vars != 2:
            return "Solo soporta 2 variables por ahora.", 400

        objetivo = request.form.get('objetivo')
        if not objetivo:
            return "Función objetivo requerida.", 400

        maximizar = (request.form.get('tipo_objetivo', 'max') == 'max')

        restricciones_texto = request.form.get('restricciones', '')
        restricciones = [line.strip() for line in restricciones_texto.strip().split('\n') if line.strip()]
        if len(restricciones) == 0:
            return "Debes ingresar al menos una restricción.", 400

        f_obj = parse_func_objetivo(objetivo, num_vars)

        A_ub, b_ub = [], []
        for r in restricciones:
            coef, val, tipo = parse_restriccion(r, num_vars)
            if tipo == '<=':
                A_ub.append(coef)
                b_ub.append(val)
            elif tipo == '>=':
                A_ub.append([-c for c in coef])
                b_ub.append(-val)
            elif tipo == '=':
                A_ub.append(coef)
                b_ub.append(val)
                A_ub.append([-c for c in coef])
                b_ub.append(-val)

        if len(A_ub) == 0:
            return "No hay restricciones válidas para resolver.", 400

        c = [-c for c in f_obj] if maximizar else f_obj

        res = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), method='highs')

        if not res.success:
            return render_template('resultado.html', resultado="No se encontró solución óptima.", imagen=None, procesos="")

        resultado = f"{'Máximo' if maximizar else 'Mínimo'} G = {-res.fun if maximizar else res.fun:.2f}, Variables = {res.x.round(2)}"

        vertices = calcular_vertices_2d(np.array(A_ub), np.array(b_ub))

        grafico_html, grafico_json = graficar_2d_plotly(np.array(A_ub), np.array(b_ub), vertices, res, maximizar)

        return render_template('resultado_interactivo.html', resultado=resultado,
                       grafico_html=grafico_html, grafico_json=grafico_json, procesos="")


    except Exception as e:
        return f"Error procesando la solicitud: {e}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render asigna PORT automáticamente
    app.run(host='0.0.0.0', port=port)

