# Auditoria de traduccion MATLAB -> Python para W1 / Fuchs

Fecha: 2026-05-12

Objetivo: revisar si la reproduccion del `W1_Warp_Shell.mlx` de WarpFactory
esta fallando por una traduccion incorrecta de MATLAB a Python, antes de
atribuir el desacuerdo a la fisica del caso subluminal.

## Resumen corto

Correccion critica posterior: la sospecha sobre `Swarp` era correcta. El port
Python de `compact_sigmoid` tenia la transicion radial invertida dentro de la
shell: devolvia `sig` donde la formula MATLAB/paper implica `1 - sig`. Eso hacia
que en `r~11.7` el shift efectivo fuera `S_eff~0.045` en vez de `S_eff~0.968`
despues de smoothing, desplazando los gradientes del shift y generando una
violacion NEC espuria.

Se corrigio `warpfactory/utils/helpers.py::compact_sigmoid` y se agrego
`tests/validate_compact_sigmoid_profile.py` y
`tests/validate_fuchs_w1_shifted_after_swarp_fix.py`. Despues de la correccion,
W1 original con `vWarp=0.02`, `christoffel`, `num_vecs=40`,
`num_time_vecs=10` queda sin violaciones sampled en la region material:

```text
NEC min = 1.365148e39, frac = 0
WEC min = 1.365148e39, frac = 0
SEC min = 9.757740e38, frac = 0
DEC min = 3.744716e39, frac = 0
```

Por tanto, los resultados negativos anteriores para el caso shifted no deben
usarse como evidencia contra el paper. Fueron evidencia util para encontrar el
bug: el perfil efectivo de shift estaba invertido.

La construccion de la metrica W1 ahora queda alineada con MATLAB en el punto
critico:

- `g_tx = -ShiftMatrix * vWarp` se copia literalmente.
- `ShiftMatrix = compactSigmoid(...)` ahora tiene la orientacion 1 -> 0
  correcta desde el interior hacia el exterior.
- No se agrega por defecto la correccion ADM `beta_i beta^i` en `g_tt`.
- La receta original usa `R1=10`, `R2=20`, `factor=1/3`, `vWarp=0.02`,
  `sigma=0`, `smoothFactor=4000`, `gridSize=[1,300,300,5]`.

El punto sospechoso principal ya no es el paper ni `doFrameTransfer`; era el
perfil radial efectivo del shift. Queda pendiente rerunear/limpiar los artefactos
historicos generados antes del fix.

Los artefactos previos al fix quedaron marcados como obsoletos en:

```text
outputs/DEPRECATED_PRE_SWARP_FIX.md
```

El punto sospechoso historico era la cadena:

```text
met2den.m
  -> ricciT.m
  -> takeFiniteDifference1.m / takeFiniteDifference2.m
  -> einT.m / einE.m
  -> doFrameTransfer.m
  -> getEnergyConditions.m
```

En los resultados ya guardados, la ruta Python independiente
`solver_method="christoffel"` reproduce la cascara estatica limpia en la region
material. La ruta literal `solver_method="warpfactory_direct"`, que deberia ser
la traduccion mas cercana de `met2den.m`, todavia no reproduce limpiamente la
misma cascara estatica. Eso acota la auditoria prioritaria a `ricciT.m` y las
derivadas finitas, no al generador `WarpShell`.

## Mapa de archivos

| MATLAB | Python | Estado |
| --- | --- | --- |
| `Metrics/WarpShell/metricGet_WarpShellComoving.m` | `warpfactory/generator/warp_shell.py` | Alineado en la construccion principal; revisar solo contra arrays exportados. |
| `Metrics/utils/compactSigmoid.m` | `warpfactory/utils/helpers.py::compact_sigmoid` | Corregido: la transicion interior debe usar `1 - sig`, no `sig`. |
| `Metrics/utils/sph2cartDiag.m` | `warpfactory/utils/helpers.py::sph2cart_diag` | Alineado; la forma Python usa `I + (B-1)n n^T`, equivalente. |
| `Metrics/utils/TOVconstDensity.m` | `warpfactory/utils/helpers.py::tov_const_density` | Pendiente de comparacion numerica directa contra `P` y `P_smooth`. |
| `Metrics/utils/alphaNumericSolver.m` | `warpfactory/utils/helpers.py::alpha_numeric_solver` | Alineado por lectura; pendiente de comparacion directa de `A`. |
| `Solver/getEnergyTensor.m` | `warpfactory/solver/solvers.py::solve_energy_tensor` | Dos rutas: `christoffel` funciona como control; `warpfactory_direct` es la ruta de paridad pendiente. |
| `Solver/utils/met2den.m` | `warpfactory/solver/tensor_utils.py` | Estructura alineada: inversa, Ricci, escalar, Einstein, energia. |
| `Solver/utils/ricciT.m` | `warpfactory/solver/tensor_utils.py::get_ricci_tensor_warpfactory_direct` | Principal sospechoso. Necesita comparacion componente por componente. |
| `Solver/utils/takeFiniteDifference1.m` | `warpfactory/solver/finite_difference.py::derive_1st_4th_order` | Alineado por lectura, incluyendo copia de bordes. |
| `Solver/utils/takeFiniteDifference2.m` | `warpfactory/solver/finite_difference.py::derive_2nd_4th_order` | Alineado por lectura; las derivadas mixtas son sensibles y deben compararse con arrays. |
| `Solver/utils/einT.m` | `warpfactory/solver/tensor_utils.py::get_einstein_tensor` | Alineado. |
| `Solver/utils/einE.m` | `warpfactory/solver/tensor_utils.py::get_energy_tensor` | Alineado por lectura; usa elevacion doble de indices. |
| `Analyzer/doFrameTransfer.m` | `warpfactory/analyzer/transform.py::do_frame_transfer` | Alineado por lectura; conviene comparar `T_hat` directamente. |
| `Analyzer/utils/getEulerianTransformationMatrix.m` | `warpfactory/analyzer/transform.py::get_eulerian_transformation_matrix` | Alineado por formula. |
| `Analyzer/changeTensorIndex.m` | `warpfactory/analyzer/matlab_compat.py::matlab_change_tensor_index` | Alineado por lectura para rank-2. |
| `Analyzer/getEnergyConditions.m` | `warpfactory/analyzer/energy_conditions.py::_evaluate_warpfactory_compatible_maps_from_eulerian` | NEC/WEC/SEC alineadas; DEC usa la convencion MATLAB de vectores null y norma de flujo. |
| `Analyzer/utils/getEvenPointsOnSphere.m` | `warpfactory/analyzer/matlab_compat.py::matlab_even_points_on_sphere` | Ya portado literalmente. |
| `Analyzer/utils/generateUniformField.m` | `warpfactory/analyzer/matlab_compat.py::matlab_generate_uniform_field` | Ya portado literalmente. |

## Hallazgos importantes

1. `metricGet_WarpShellComoving.m` y `create_warp_shell_metric` coinciden en la
   parte critica del shift:

   ```text
   MATLAB: Metric.tensor{1,2} = Metric.tensor{1,2} - Metric.tensor{1,2}.*ShiftMatrix - ShiftMatrix*vWarp
   Python: tensor[0,1] = -shift_val * v_warp
   ```

   Como `Metric.tensor{1,2}` parte de cero en MATLAB, ambas expresiones dan el
   mismo `g_tx`.

2. La opcion Python `adm_shift_g00=True` existe, pero el valor por defecto es
   `False`, que es lo correcto para paridad con WarpFactory original.

3. La ruta `christoffel` da el control fisico mas estable para la cascara
   estatica material. La ruta `warpfactory_direct` es la que debe igualar a
   MATLAB, pero sus resultados guardados muestran WEC/DEC negativos en la
   cascara estatica. Eso significa que no podemos usarla todavia como prueba
   de Fuchs sin auditar `ricciT` y derivadas.

4. Los resultados shifted negativos previos se explican por el `Swarp` invertido:
   la metrica generada no era la receta W1 pretendida. Despues del fix, el caso
   `vWarp=0.02` pasa las condiciones sampled en la region material con
   `num_vecs=40`.

## Entorno local de verificacion

En la instalacion actual de Codex, el PATH inicial no trae Python/conda/gcloud,
pero las herramientas estan instaladas. El entorno usable para pyWarpFactory es:

- Python: `C:\Users\Nelson\anaconda3\envs\astra\python.exe`
- gcloud: `C:\Users\Nelson\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd`
- Proyecto gcloud activo: `warpopt`
- Zona gcloud activa: `us-west1-a`
- Cuenta activa: `astrumdrivetechnologies@astrumdrive.com`

El Python base de Anaconda no debe usarse para este proyecto en esta maquina:
puede disparar un error de Windows por falta de `zlib.dll`. El entorno `astra`
si contiene `zlib.dll` y corre las validaciones.

Para reconstruir el PATH en una terminal PowerShell sin tocar variables globales
de Windows:

```powershell
.\tools\use_local_paths.ps1
```

Validaciones ya corridas con `astra`:

- `tests\validate_matlab_compat.py`
- `tests\validate_warp_shell_components.py`
- `tests\validate_fuchs_recipe.py`
- `tests\validate_w1_reference_tools.py`
- `tests\validate_cloud_job_config.py`

## Siguiente prueba minima

La siguiente comparacion no debe empezar por las condiciones de energia finales.
Debe comparar componentes intermedias en este orden:

1. `MetricTensor`
2. `EnergyTensor` antes de `doFrameTransfer`
3. `EnergyEulerianTensor`
4. `NullMap`, `WeakMap`, `StrongMap`, `DominantMap`

Los scripts de auditoria ya guardan/comparan ese `EnergyTensor` intermedio:

- MATLAB: `tools/matlab_export_w1_component_audit.m`
- Python: `Examples/audit_fuchs_w1_components.py --save-arrays`
- Comparador: `tools/compare_w1_component_audit.py`

Si MATLAB esta disponible fuera de este shell:

```matlab
run('tools/matlab_export_w1_component_audit.m')
```

Despues, con Python disponible:

```powershell
python tools\convert_w1_mat_to_npz.py `
  tools\w1_component_audit_reference.mat `
  --output tools\w1_component_audit_reference.npz

python tools\compare_w1_component_audit.py `
  outputs\fuchs_w1_component_audit_original_arrays\fuchs_w1_component_audit_arrays.npz `
  tools\w1_component_audit_reference.npz
```

Si `MetricTensor` coincide pero `EnergyEulerianTensor` no, el error esta antes
o dentro de `doFrameTransfer`. Si `EnergyTensor` ya no coincide, el error esta
en `met2den`/`ricciT`/derivadas. Si ambos tensores coinciden pero los mapas no,
el error esta en `getEnergyConditions`.

## Matriz operativa de paridad

Esta matriz define que debe estar congelado antes de afirmar que pyWarpFactory
reproduce el WarpFactory MATLAB usado por el paper.

| Bloque | MATLAB | Python | Prueba actual | Criterio de paridad | Estado |
| --- | --- | --- | --- | --- | --- |
| Derivadas 4to orden | `takeFiniteDifference1.m`, `takeFiniteDifference2.m` | `warpfactory/solver/finite_difference.py` | `tests/validate_matlab_finite_difference_parity.py` | Igualdad exacta contra una referencia literal de las formulas MATLAB, incluyendo bordes y `phiphiFlag`. | Cerrado localmente; `phiphiFlag` corregido en Python. |
| Inversa metrica | `c4Inv.m` | `get_c4_inv` | `tests/validate_matlab_linear_algebra_parity.py` | `g_{mu alpha} g^{alpha nu} = delta_mu^nu` para metricas sinteticas no diagonales, con ejes `4x4+grid` correctos. | Cerrado localmente para convencion de ejes; falta referencia MATLAB externa si se quiere bit-level. |
| Ricci directo | `ricciT.m` | `get_ricci_tensor_warpfactory_direct` | `tests/validate_warpfactory_direct_ricci.py`; export W1 ampliado | Coincidencia componente por componente de `R_munu` o, al menos, primer desacuerdo localizado en `diff_1_gl`/`diff_2_gl`. | Pasa casos planos no triviales; sigue siendo el principal sospechoso para paridad W1. |
| Einstein tensor | `einT.m` | `get_einstein_tensor` | `tests/validate_matlab_einstein_energy_parity.py` | `G_mn = R_mn - 0.5 g_mn R` igual dado el mismo `R_mn`. | Pasa caso sintetico no diagonal. |
| Energia contravariante | `einE.m` | `get_energy_tensor` | `tests/validate_matlab_einstein_energy_parity.py` | `T^{mu nu}` igual dado el mismo `G_mn` y `g^{mu nu}`. | Pasa caso sintetico no diagonal; sensible a escala `c^4/G`. |
| Frame Euleriano | `doFrameTransfer.m`, `getEulerianTransformationMatrix.m` | `do_frame_transfer`, `get_eulerian_transformation_matrix` | `tests/validate_frame_transfer.py`, `tests/validate_frame_transfer_known_metrics.py`, `tests/validate_matlab_compat.py` | `M^T g M = eta`, roundtrip de indices, y transformacion manual de tensores completos con shift constante. | Pasa casos sinteticos; requiere comparacion W1. |
| Condiciones de energia | `getEnergyConditions.m` | `energy_conditions.py`, `matlab_compat.py` | `tests/validate_energy_condition_methods.py`, `tests/validate_energy_conditions_known_tensors.py`, `tests/validate_matlab_compat.py` | Mismos mapas NEC/WEC/SEC/DEC dado el mismo `T_hat` y mismos observadores. | Pasa fluidos ideales sinteticos; DEC MATLAB tiene una salvedad importante. |
| Metrica W1 | `metricGet_WarpShellComoving.m` | `recipes/fuchs_warp_shell.py`, `generator/warp_shell.py` | `tests/validate_warp_shell_components.py`, `tests/validate_fuchs_recipe.py` | Coincidencia de `MetricTensor` para receta original quick/original. | Lectura alineada; falta referencia MATLAB externa. |

Orden de ataque desde este punto:

1. Validar `c4Inv`/`get_c4_inv` en un caso sintetico no diagonal.
2. Exportar o reconstruir intermedios `diff_1_gl` y `diff_2_gl` para W1 quick.
3. Comparar `R_munu` entre `ricciT.m` y `get_ricci_tensor_warpfactory_direct`.
4. Comparar `RicciScalar`, `EinsteinTensor` y `EnergyTensor`.
5. Solo despues comparar `EnergyEulerianTensor` y mapas.

Nota: la correccion de `phiphiFlag` no explica directamente el fallo W1 shifted,
porque `ricciT.m` fija `phiphiFlag = 0`. Se corrigio porque era una discrepancia
literal y puede afectar otros analizadores como `covDiv.m`.

### Estado Ricci directo

Se agrego `tests/validate_warpfactory_direct_ricci.py` con tres espaciotiempos
planos:

- Minkowski cartesiano.
- Minkowski con boost constante y termino fuera de diagonal.
- Rindler, donde `g00` tiene derivadas primeras y segundas no nulas pero la
  curvatura fisica sigue siendo cero.

La ruta `get_ricci_tensor_warpfactory_direct` da Ricci maximo interior
`~9e-15` en Rindler. Esto descarta un error grueso en la formula directa. No
descarta una divergencia MATLAB/Python especifica de W1, especialmente en
bordes, escalas temporales, convenciones de indices o en la composicion completa
`ricciT -> ricciS -> einT -> einE`.

El export/comparador W1 ahora incluye `RicciTensor`:

- MATLAB: `tools/matlab_export_w1_component_audit.m`
- Python: `Examples/audit_fuchs_w1_components.py --save-arrays`
- Conversor: `tools/convert_w1_mat_to_npz.py`
- Comparador: `tools/compare_w1_component_audit.py`

Tambien incluye `RicciScalar` y `EinsteinTensor`, de modo que la cadena completa
queda separada en estos puntos de corte:

```text
MetricTensor
  -> RicciTensor
  -> RicciScalar
  -> EinsteinTensor
  -> EnergyTensor
  -> EnergyEulerianTensor
  -> Null/Weak/Strong/Dominant maps
```

Se agrego `tests/validate_matlab_einstein_energy_parity.py`, que compara
`ricciS`, `einT` y `einE` contra referencias de bucles literales estilo MATLAB
en una metrica sintetica no diagonal. La prueba pasa con tolerancia relativa
`1e-13` en `EnergyTensor`, necesaria por la escala `c^4/G`.

Nueva prueba Python quick generada localmente:

```powershell
python Examples\audit_fuchs_w1_components.py `
  --profile quick `
  --case both `
  --num-vecs 4 `
  --critical-points 2 `
  --solver-method warpfactory_direct `
  --output-dir outputs\codex_ricci_audit_quick `
  --save-arrays
```

El archivo resultante
`outputs/codex_ricci_audit_quick/fuchs_w1_component_audit_arrays.npz` contiene
`static_ricci_tensor` y `shifted_ricci_tensor`, ademas de metricas, tensores de
energia y mapas de condiciones.

Prueba actualizada con los puntos de corte adicionales:

```powershell
python Examples\audit_fuchs_w1_components.py `
  --profile quick `
  --case both `
  --num-vecs 4 `
  --critical-points 2 `
  --solver-method warpfactory_direct `
  --output-dir outputs\codex_curvature_audit_quick `
  --save-arrays
```

El archivo
`outputs/codex_curvature_audit_quick/fuchs_w1_component_audit_arrays.npz`
contiene `static_ricci_scalar`, `static_einstein_tensor`,
`shifted_ricci_scalar` y `shifted_einstein_tensor`.

## Estado frame y condiciones de energia

Se agrego `tests/validate_frame_transfer_known_metrics.py` para separar dos
riesgos:

- La tetrada Euleriana debe satisfacer `M^T g M = eta`.
- La conversion de indices y la transformacion `M^T T_cov M`, con flip de
  signos en componentes `0i`, debe coincidir con una transformacion manual.

La prueba usa Minkowski y una metrica con shift constante:

```text
g = [[-1 + beta^2, beta, 0, 0],
     [ beta,       1,    0, 0],
     [ 0,          0,    1, 0],
     [ 0,          0,    0, 1]]
```

Tambien se agrego `tests/validate_energy_conditions_known_tensors.py`, con
fluidos isotropicos en marco Euleriano:

- polvo positivo: todas las condiciones no negativas;
- radiacion isotropica: todas no negativas;
- energia negativa: NEC/WEC/SEC negativas;
- presion mayor que densidad: DEC compatible MATLAB negativa;
- constante cosmologica: SEC negativa, NEC/WEC en cero numerico.

Hallazgo importante: el branch MATLAB/WarpFactory de `Dominant` no implementa por
si solo una DEC completa en el sentido `WEC + flujo causal futuro`. En el codigo
original, DEC construye un flujo con vectores nulllike, evalua su norma
Minkowski y luego invierte el signo para que negativo signifique violacion. Por
eso una energia negativa pura es detectada por NEC/WEC/SEC, pero no
necesariamente por ese mapa `Dominant`. Para reproducir figuras WarpFactory hay
que respetar esa convencion; para interpretar fisicamente DEC conviene mirar
`Dominant` junto con WEC/NEC o usar la ruta `standard`.

## Auditoria de observadores criticos

`Examples/audit_fuchs_w1_components.py` ahora guarda, desde el mismo
`EnergyEulerianTensor`, dos familias de mapas:

- `*_standard_null`, `*_standard_weak`, `*_standard_strong`,
  `*_standard_dominant`
- `*_warpfactory_null`, `*_warpfactory_weak`, `*_warpfactory_strong`,
  `*_warpfactory_dominant`

El JSON tambien incluye `critical_observers`, que para cada metodo y condicion
guarda los puntos mas negativos de la region `material_shell`, el vector que
produce la contraccion minima, el subcriterio usado y componentes locales de
`T_hat`. Para contracciones cuadraticas guarda ademas:

```text
diag = T00 k0^2 + T11 k1^2 + T22 k2^2 + T33 k3^2
offdiag = sum_{a!=b} Tab ka kb
total = diag + offdiag
```

Para `Dominant` guarda tambien el flujo `J`, `J.J`, `-J.J` y la norma firmada
usada por la rama WarpFactory.

Prueba local:

```powershell
python Examples\audit_fuchs_w1_components.py `
  --profile quick `
  --case shifted `
  --num-vecs 4 `
  --num-time-vecs 5 `
  --critical-points 2 `
  --solver-method warpfactory_direct `
  --output-dir outputs\codex_observer_audit_quick `
  --save-arrays
```

Artefactos:

- `outputs/codex_observer_audit_quick/fuchs_w1_component_audit.json`
- `outputs/codex_observer_audit_quick/fuchs_w1_component_audit_arrays.npz`

Resultado quick shifted en `material_shell`:

```text
standard Null      min=-1.492754343562337e41, frac=0.29466553767993225
warpfactory Null   min=-6.653674032119489e40, frac=0.31879762912785775
warpfactory Weak   min=-6.653674032119489e40, frac=0.4847586790855207
warpfactory Strong min=-6.653674032119489e40, frac=0.31879762912785775
warpfactory DEC    min=-6.449356857668349e40, frac=0.8433530906011855
```

En el punto critico WarpFactory NEC quick:

```text
idx=[0, 38, 24, 4], r=10.32
T00=-4.266069482663569e39
T01=-7.897473353643737e40
vector=[0.7071067811865475, -0.5048419596572157,
        -0.4624765894285484, 0.17677669529663687]
diag= 2.8899e39
offdiag=-6.9427e40
total=-6.6537e40
```

Interpretacion: el valor negativo no parece venir de ruido numerico pequeno. El
termino de flujo `T01` es casi veinte veces mayor que `T00` en magnitud y el
observador nulo critico tiene direccion espacial que lo acopla fuertemente. Esto
mantiene el foco en el sector de momento/flujo del tensor Euleriano. La siguiente
comparacion contra MATLAB debe mirar si `EnergyEulerianTensor(0,1)` coincide en
esos puntos; si coincide, el desacuerdo con el paper es de parametros/receta o de
criterio de evaluacion, no de traduccion del mapa.

Verificacion adicional sobre la corrida Compute original
`outputs/fuchs-original-shift-20260512-arrays.npz`:

```text
idx=(0,150,91,2), r=11.700427342623007
coords=(x=0.1, y=-11.7, z=0.0)
NEC map=-1.301711061879728e40
T00= 1.3870828173485198e40
T01=-1.8749597049495609e40
T11=-2.792058011704236e39
T22=-3.981492095151964e38
T33= 1.724808413708196e39
critical vector=[0.7071067811865475, -0.700961665020447,
                 -0.028986965549406697, 0.08838834764831852]
```

Descomposicion de la contraccion en ese punto:

```text
T_ab k^a k^b = -1.3017110618797279e40
diag contribution    =  5.576684567757786e39
offdiag contribution = -1.8593795186555062e40
```

En la corrida original la densidad Euleriana `T00` es positiva en el punto
critico, pero la contribucion fuera de diagonal, dominada por `T01`, vuelve la
NEC negativa. Esto refuerza que el fallo shifted esta en el sector
momento/flujo o en su generacion/transformacion, no en un mapa NEC ruidoso.

## Barrido de velocidad y slices W1

Se agrego `Examples/sweep_fuchs_w1_velocity_slices.py` para barrer `vWarp` sobre
la receta W1, con resumen global, region `material_shell`, slice central en `z`
y lineas centrales. La herramienta tambien reporta magnitudes Eulerianas como
`T01_absmax` y `rho_min`.

Comandos corridos:

```powershell
python Examples\sweep_fuchs_w1_velocity_slices.py `
  --profile original `
  --velocities 0,0.005,0.01,0.02 `
  --solver-methods christoffel `
  --num-vecs 4 `
  --num-time-vecs 5 `
  --output outputs\fuchs_w1_velocity_slices_original_christoffel.json

python Examples\sweep_fuchs_w1_velocity_slices.py `
  --profile original `
  --velocities 0.006,0.007,0.008,0.009 `
  --solver-methods christoffel `
  --num-vecs 4 `
  --num-time-vecs 5 `
  --output outputs\fuchs_w1_velocity_threshold_original_christoffel.json
```

Resultados `profile=original`, solver `christoffel`, metodo `warpfactory`,
region `material_shell`:

```text
v=0       NEC_min= 1.8223994832080958e39, frac=0.0
v=0.005   NEC_min= 1.0545490119378651e39, frac=0.0
v=0.006   NEC_min= 3.150625318064236e38,  frac=0.0
v=0.007   NEC_min=-4.266312895563969e38,  frac=0.001782319391634981
v=0.008   NEC_min=-1.167708894640002e39,  frac=0.01872284084736556
v=0.009   NEC_min=-1.9081680180419736e39, frac=0.03534084736556219
v=0.01    NEC_min=-2.648006398839329e39,  frac=0.0504311515480717
v=0.02    NEC_min=-1.0011755190908808e40, frac=0.27955255296034764
```

Con `num_vecs=4`, el umbral sampled NEC para esta ruta queda entre
`vWarp=0.006` y `vWarp=0.007`, muy por debajo del `vWarp=0.02` citado en
W1/paper. Este umbral fue refinado despues con `num_vecs=40`: ver la seccion
"Tabla final de evidencia W1", donde el primer fallo sampled aparece entre
`vWarp=0.004` y `vWarp=0.005`.

Resumen de slices:

```text
v=0.02 shell NEC=-1.0011755190908808e40, frac=0.27955
v=0.02 central-z NEC=-1.0011420964895533e40, frac=0.27911
v=0.02 central xz-line NEC= 1.8254827648572105e39, frac=0.0
v=0.02 central yz-line NEC=-7.602944718789568e39, frac=0.4
```

Esto confirma que algunos cortes/lineas pueden parecer fisicos mientras otros
cortes del mismo volumen no lo son. En particular, el resultado depende de que
slice se mire, pero la region 3D material no pasa para `vWarp=0.02`.

Tambien se probo la convencion ADM experimental `g00 += beta_i beta^i`:

```powershell
python Examples\sweep_fuchs_w1_velocity_slices.py `
  --profile original `
  --velocities 0.02 `
  --solver-methods christoffel `
  --num-vecs 4 `
  --num-time-vecs 5 `
  --adm-shift-g00 `
  --output outputs\fuchs_w1_velocity_original_christoffel_adm_shift.json
```

Resultado:

```text
v=0.02, adm_shift_g00=True:
shell NEC=-1.0424e40, frac=0.2779
```

La correccion ADM de `g00` no elimina la violacion. El flujo `T01` sigue siendo
la escala dominante.

## Corrida Compute original shifted 2026-05-12

Se lanzo una corrida pesada en Google Compute con la receta W1 original y shift
activado:

```powershell
.\tools\run_compute_job.ps1 `
  -ProjectId warpopt `
  -Zone us-west1-a `
  -InstanceName pywarpfactory-fuchs-original-shift-20260512 `
  -MachineType e2-standard-8 `
  -Profile original `
  -VWarp 0.02 `
  -NumVecs 40 `
  -OutputUri gs://warpopt-data/pywarpfactory/runs/fuchs-original-shift-20260512 `
  -SaveArrays
```

La VM termino en estado `TERMINATED` y subio:

- `gs://warpopt-data/pywarpfactory/runs/fuchs-original-shift-20260512/summary.json`
- `gs://warpopt-data/pywarpfactory/runs/fuchs-original-shift-20260512/arrays.npz`
- `gs://warpopt-data/pywarpfactory/runs/fuchs-original-shift-20260512/compute-startup.log`

Copias locales:

- `outputs/fuchs-original-shift-20260512-summary.json`
- `outputs/fuchs-original-shift-20260512-arrays.npz`
- `outputs/fuchs-original-shift-20260512-compute-startup.log`

Resultado global:

- NEC: minimo `-1.4558670830183307e40`, fraccion violada `0.7710022222222223`
- WEC: minimo `-1.4558670830183307e40`, fraccion violada `0.7711666666666667`
- SEC: minimo `-1.4558670830183305e40`, fraccion violada `0.77284`
- DEC: minimo `-1.4888779175819598e40`, fraccion violada `0.24122666666666667`

Resultado en la region `material_shell`:

- NEC: minimo `-1.301711061879728e40`, fraccion violada `0.3313416621401412`
- WEC: minimo `-1.301711061879728e40`, fraccion violada `0.3313416621401412`
- SEC: minimo `-1.301711061879728e40`, fraccion violada `0.33665467137425314`
- DEC: minimo `-1.4888779175819598e40`, fraccion violada `0.33303910917979357`

El archivo `arrays.npz` contiene arrays finitos para `metric_tensor`,
`energy_tensor`, `eulerian_energy_tensor`, `null`, `weak`, `strong` y
`dominant`. Por tanto, la corrida no fallo por instalacion ni por NaNs: confirma
el fallo fisico/numerico del caso shifted en la ruta Python actual.

## Reorientacion sin MATLAB: paper-only + geometry-only

Como no tenemos MATLAB disponible, la comparacion contra WarpFactory original ya
no debe bloquear la auditoria principal. Desde este punto quedan congelados dos
motores:

- `christoffel`: motor fisico independiente principal.
- `warpfactory_direct`: motor legacy de compatibilidad MATLAB, util para
  auditoria historica, pero no como verdad fisica mientras no reproduzca
  limpiamente la shell estatica.

Se agrego `tests/validate_christoffel_physics.py` como validacion independiente
del motor `christoffel`. Prueba:

- Minkowski cartesiano: `Ricci = 0`, `T = 0`.
- Minkowski con shift ADM constante: `Ricci = 0`, `T = 0`.
- Minkowski con boost constante: `Ricci = 0`, `T = 0`.
- Rindler: curvatura fisica cero, con piso finito de diferencias numericas.
- Schwarzschild exterior: `Ricci ~ 0` fuera de la zona fuerte.
- W1 estatica original, `vWarp=0`: NEC/WEC/SEC/DEC positivas en region material.
- Nucleo pasajero central: derivadas espaciales de la metrica y `T` cercanas a
  cero.

Corrida local:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  tests\validate_christoffel_physics.py
```

Resultado clave:

```text
Minkowski / shift constante / boost constante: Ricci=0, T=0
Rindler: max |Ricci|=1.1478e-04, max |R|=2.2899e-04
Schwarzschild exterior: max |Ricci|=2.0900e-05, max |R|=6.3606e-05
Static W1 original shell:
  Null min=1.8224e39, frac=0
  Weak min=1.8224e39, frac=0
  Strong min=9.9695e38, frac=0
  Dominant min=3.8227e39, frac=0
Passenger core r<2:
  max |spatial dg|=1.5266e-15
  max |T|=4.8092e28
  max |T_hat|=4.5138e28
```

Nota importante: el perfil `quick` no debe usarse como prueba fisica de la shell
estatica, porque su malla/smoothing tosco ya introduce una pequena violacion
NEC estatica. Para validaciones fisicas de W1 hay que usar `profile=original`.

## Export paper-style y certificados NEC

Se agrego `Examples/export_fuchs_w1_paper_regression.py` para producir datos
directamente comparables con las figuras del paper sin depender de MATLAB:

- Fig. 8: perfiles sobre eje `y` de `g00`, `g01`, `g11`, `g22`, `g33`.
- Fig. 9: perfiles sobre eje `y` de `rho`, flujos, presiones principales,
  presiones radial/tangencial y anisotropia.
- Fig. 10: perfiles sobre eje `y` de NEC/WEC/SEC/DEC.
- Fig. 14: mapas 2D en el slice central de `z` para NEC/WEC/SEC/DEC.
- Certificados locales NEC para los puntos mas negativos de la region material:
  coordenadas, metrica local, `T_hat`, vector nulo critico, `eta(k,k)`, `Tkk`,
  y descomposicion `diag/offdiag`.

Corrida original local:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\export_fuchs_w1_paper_regression.py `
  --profile original `
  --v-warp 0.02 `
  --num-vecs 4 `
  --num-time-vecs 5 `
  --critical-points 10 `
  --output-dir outputs\codex_paper_regression_original_v002
```

Artefactos:

- `outputs/codex_paper_regression_original_v002/fuchs_w1_paper_regression_arrays.npz`
- `outputs/codex_paper_regression_original_v002/fuchs_w1_paper_regression_summary.json`

Resultado material shell, `christoffel`, metodo `warpfactory`, `num_vecs=4`:

```text
NEC min=-1.0011755190908808e40, frac=0.27955255296034764
WEC min=-1.0011755190908808e40, frac=0.27955255296034764
SEC min=-1.0011755190908805e40, frac=0.2908066268332428
DEC min=-1.4712242682162943e40, frac=0.3112099402498642
```

Primer certificado NEC en region material:

```text
coords=(x=-3.9, y=9.5, z=-0.4), r~10.276
Tkk=-1.0011755190908808e40
diag= 4.9326940352772046e39
offdiag=-1.4944449226186012e40
```

Esto reproduce el patron ya visto: la contribucion diagonal es positiva, pero
los terminos fuera de diagonal dominados por el sector de flujo vuelven negativa
la contraccion NEC. La magnitud esta muchos ordenes por encima del piso numerico
del nucleo pasajero y del piso de curvatura observado en Rindler.

## Nuevo arbol de decision

La auditoria queda ahora asi:

1. Si los perfiles metricos paper-style no coinciden con las figuras, revisar
   smoothing exacto, centro/radio, permutacion de ejes, si el paper grafica
   `g01`, `beta_i`, `beta^i` o una cantidad escalada, y factores `c`/`c^2`.
2. Si los perfiles metricos coinciden pero las condiciones de energia no,
   usar los certificados NEC y repetir con mas observadores, mas resolucion,
   otro orden de diferencias finitas y, si es posible, derivadas radiales
   semi-analiticas/autodiff.
3. Si la violacion persiste muy por encima del piso numerico, clasificar el
   resultado como:

```text
published recipe / displayed result not reproduced by independent curvature evaluation
```

No llamarlo todavia "paper wrong". Las posibilidades abiertas son: receta real
no publicada exactamente igual, smoothing/interpolacion insuficientemente
especificados, figura basada en slice/sampling que no ve el observador critico,
convencion/bug en WarpFactory, o afirmacion fisica no sostenida para la metrica
publicada.

## Tabla final de evidencia W1

Se agrego `Examples/fuchs_w1_final_evidence_table.py` para producir una tabla
compacta por velocidad con:

- minimos NEC/WEC/SEC/DEC en `material_shell`;
- fracciones violadas;
- `max |T01|` y `min T00`;
- punto critico NEC;
- `Tkk`, `diag` y `offdiag` del certificado local.

Corrida base con 4 direcciones nulas:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\fuchs_w1_final_evidence_table.py `
  --profile original `
  --velocities 0,0.005,0.006,0.007,0.01,0.02 `
  --num-vecs 4 `
  --num-time-vecs 5 `
  --output-dir outputs\codex_final_evidence_original_n4
```

Artefactos:

- `outputs/codex_final_evidence_original_n4/fuchs_w1_final_evidence_table.json`
- `outputs/codex_final_evidence_original_n4/fuchs_w1_final_evidence_table.csv`
- `outputs/codex_final_evidence_original_n4/fuchs_w1_final_evidence_table.md`

Resumen `num_vecs=4`:

| vWarp | NEC min | NEC frac | DEC min | max \|T01\| | min T00 | r critico | Tkk | diag | offdiag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.822399e39 | 0.000000 | 3.822713e39 | 0 | 5.872160e39 | 19.999500 | 1.822399e39 | 2.030613e39 | -2.082137e38 |
| 0.005 | 1.054549e39 | 0.000000 | 2.314565e39 | 4.687403e39 | 5.867622e39 | 10.253780 | 1.054549e39 | 4.820134e39 | -3.765585e39 |
| 0.006 | 3.150625e38 | 0.000000 | -5.956402e38 | 5.624883e39 | 5.865626e39 | 10.277159 | 3.150625e38 | 4.853106e39 | -4.538044e39 |
| 0.007 | -4.266313e38 | 0.001782 | -3.049918e39 | 6.562363e39 | 5.863266e39 | 10.277159 | -4.266313e38 | 4.855949e39 | -5.282581e39 |
| 0.01 | -2.648006e39 | 0.050431 | -6.860968e39 | 9.374804e39 | 5.854011e39 | 10.277159 | -2.648006e39 | 4.867102e39 | -7.515109e39 |
| 0.02 | -1.001176e40 | 0.279553 | -1.471224e40 | 1.874960e40 | 5.790747e39 | 10.277159 | -1.001176e40 | 4.932694e39 | -1.494445e40 |

Luego se repitio con 40 direcciones nulas y 10 shells timelike:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\fuchs_w1_final_evidence_table.py `
  --profile original `
  --velocities 0.006,0.007,0.02 `
  --num-vecs 40 `
  --num-time-vecs 10 `
  --output-dir outputs\codex_final_evidence_original_n40
```

Resultado `num_vecs=40`:

| vWarp | NEC min | NEC frac | DEC min | max \|T01\| | min T00 | r critico | Tkk | diag | offdiag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.006 | -7.588777e38 | 0.033491 | -2.671021e39 | 5.624883e39 | 5.865626e39 | 19.663672 | -7.588777e38 | 2.563132e39 | -3.322010e39 |
| 0.007 | -1.363478e39 | 0.108195 | -4.100428e39 | 6.562363e39 | 5.863266e39 | 10.319884 | -1.363478e39 | 4.746524e39 | -6.110002e39 |
| 0.02 | -1.301711e40 | 0.331342 | -1.488878e40 | 1.874960e40 | 5.790747e39 | 11.700427 | -1.301711e40 | 5.576685e39 | -1.859380e40 |

Como `v=0.006` ya falla con mas observadores, se amplio el umbral:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\fuchs_w1_final_evidence_table.py `
  --profile original `
  --velocities 0.004,0.005,0.006 `
  --num-vecs 40 `
  --num-time-vecs 10 `
  --output-dir outputs\codex_final_evidence_threshold_n40
```

Resultado:

| vWarp | NEC min | NEC frac | DEC min | max \|T01\| | min T00 | r critico | Tkk | diag | offdiag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.004 | 3.525975e38 | 0.000000 | 1.767141e39 | 3.749922e39 | 5.869256e39 | 19.686036 | 3.525975e38 | 2.544221e39 | -2.191624e39 |
| 0.005 | -2.005303e38 | 0.004464 | -1.386894e39 | 4.687403e39 | 5.867622e39 | 19.686036 | -2.005303e38 | 2.540469e39 | -2.740999e39 |
| 0.006 | -7.588777e38 | 0.033491 | -2.671021e39 | 5.624883e39 | 5.865626e39 | 19.663672 | -7.588777e38 | 2.563132e39 | -3.322010e39 |

Con `num_vecs=40`, el umbral sampled NEC esta entre `vWarp=0.004` y
`vWarp=0.005`, no entre `0.006` y `0.007`. Por tanto, el valor publicado
`vWarp=0.02` queda aproximadamente cuatro veces por encima del primer umbral
sampled observado con esta evaluacion independiente.

## Figuras paper-style PNG

El exportador `Examples/export_fuchs_w1_paper_regression.py` ahora acepta
`--save-plots` y guarda:

- `fig8_metric_y_axis.png`
- `fig9_stress_y_axis.png`
- `fig10_energy_conditions_y_axis.png`
- `fig14_energy_conditions_central_z.png`

Corrida original con 40 direcciones:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\export_fuchs_w1_paper_regression.py `
  --profile original `
  --v-warp 0.02 `
  --num-vecs 40 `
  --num-time-vecs 10 `
  --critical-points 10 `
  --output-dir outputs\codex_paper_regression_original_v002_n40 `
  --save-plots
```

Resultado:

```text
Material shell NEC min=-1.3017e40
negative_fraction=0.33134
```

Artefactos:

- `outputs/codex_paper_regression_original_v002_n40/fuchs_w1_paper_regression_arrays.npz`
- `outputs/codex_paper_regression_original_v002_n40/fuchs_w1_paper_regression_summary.json`
- `outputs/codex_paper_regression_original_v002_n40/fig8_metric_y_axis.png`
- `outputs/codex_paper_regression_original_v002_n40/fig9_stress_y_axis.png`
- `outputs/codex_paper_regression_original_v002_n40/fig10_energy_conditions_y_axis.png`
- `outputs/codex_paper_regression_original_v002_n40/fig14_energy_conditions_central_z.png`

La evidencia se fortalece con mas observadores: `vWarp=0.02` no mejora, sino que
el minimo NEC baja de `-1.0012e40` a `-1.3017e40`, y la fraccion material
violada sube de `0.2796` a `0.3313`.

## Auditoria local frame/tensor en puntos criticos

Nota posterior al fix de `compact_sigmoid`: esta auditoria sigue siendo valida
como prueba de que frame/tensor no eran el bug, pero los valores negativos
mostrados en esta seccion corresponden a la metrica con `Swarp` invertido.
Despues de corregir `Swarp`, el mismo punto fisico `r~11.7004` queda con
`Tkk > 0`; ver la seccion "Correccion de Swarp y reruns".

Como la discrepancia con el paper es grande, se volvio a tratar el resultado
como posible bug de convencion. El foco inmediato fue el triangulo:

```text
shift convention + tensor index convention + Eulerian frame transfer
```

Se agrego `Examples/audit_w1_critical_frame_tensor.py`, que en un punto fisico
W1:

1. extrae `g_cov` y `g_inv`;
2. calcula `Ricci_cov`, `RicciScalar`, `Einstein_cov`;
3. calcula directamente `T_cov = c^4/(8*pi*G) * G_cov`;
4. compara ese `T_cov` contra el `T_cov` obtenido bajando indices desde el
   `T_contra` del pipeline;
5. construye la tetrada Euleriana `M`;
6. verifica `M^T g M = eta`;
7. calcula manualmente `T_hat_lower = M^T T_cov M`;
8. aplica el flip Minkowski para `T_hat_upper`;
9. compara contra `do_frame_transfer`;
10. evalua el mismo observador nulo critico y su `diag/offdiag`.

Punto critico de la corrida Compute/original:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\audit_w1_critical_frame_tensor.py `
  --profile original `
  --v-warp 0.02 `
  --target 0.1,-11.7,0.0 `
  --num-vecs 40 `
  --output-dir outputs\codex_w1_critical_frame_tensor_audit
```

Resultado:

```text
coords=(0.1, -11.7, 0), r=11.7004
M^T g M - eta absmax = 2.2204e-16
T_cov direct-vs-pipeline absmax = 2.4179e24
T_hat manual-vs-pipeline absmax = 4.8357e24
eta(k,k) = -5.5511e-17
Tkk pipeline = -1.3017e40
Tkk manual   = -1.3017e40
manual diag=5.5767e39 offdiag=-1.8594e40
```

Las diferencias absolutas `~1e24` parecen grandes solo por la escala SI; frente
a tensores `~1e40` son error relativo de redondeo `~1e-16`. En este punto:

- la tetrada Euleriana es ortonormal a precision maquina;
- bajar/subir indices del tensor no introduce el `T01` grande;
- `do_frame_transfer` coincide con la transformacion manual;
- el `Tkk < 0` se reproduce manualmente con el mismo vector nulo.

Reporte de shift local en ese punto:

```text
g01 = beta_cov_x = -9.073116167245335e-04
beta^x = -9.073084326164277e-04
beta^y = -3.7254064838315265e-07
beta_i beta^i = 8.232114808650135e-07
T_hat00 = 1.3870828173485193e40
T_hat01 = -1.8749597049495609e40
```

Tambien se probo un punto critico cerca del borde interno:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\audit_w1_critical_frame_tensor.py `
  --profile original `
  --v-warp 0.02 `
  --target=-3.9,9.5,-0.4 `
  --num-vecs 40 `
  --output-dir outputs\codex_w1_critical_frame_tensor_audit_inner
```

Resultado:

```text
coords=(-3.9, 9.5, -0.4), r=10.2772
M^T g M - eta absmax = 2.2204e-16
T_cov direct-vs-pipeline absmax = 2.4179e24
T_hat manual-vs-pipeline absmax = 2.4179e24
eta(k,k) = -5.5511e-17
Tkk pipeline = -1.0947e40
Tkk manual   = -1.0947e40
manual diag=4.8532e39 offdiag=-1.5800e40
```

Conclusion local: el fallo NEC no parece venir de `do_frame_transfer`, de la
tetrada Euleriana, ni de una confusion inmediata `T_cov`/`T_contra` dentro del
pipeline. El foco vivo se desplaza a:

1. convencion del shift publicada/implementada: `g01` vs `beta_i` vs `beta^i`;
2. perfil `Swarp`, smoothing, buffer y posicion de transiciones;
3. derivadas/interpolacion radial en la zona de shell.

La siguiente auditoria propuesta era comparar variantes de shift. Antes de
llegar a esa etapa, la auditoria de `Swarp` encontro la inversion del perfil.

## Correccion de Swarp y reruns

La funcion `warpfactory/utils/helpers.py::compact_sigmoid` tenia esta logica
incorrecta dentro de la transicion:

```text
f = sig
```

pero la formula MATLAB comentada en el mismo archivo equivale a:

```text
f = abs(sig * mask + (r >= upper) - 1)
```

Por tanto, dentro de la transicion `lower < r < upper`, debe ser:

```text
f = 1 - sig
```

Se corrigio el port y se agrego:

- `tests/validate_compact_sigmoid_profile.py`
- `tests/validate_fuchs_w1_shifted_after_swarp_fix.py`
- `Examples/audit_w1_shift_profile.py`

Validacion del perfil crudo:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  tests\validate_compact_sigmoid_profile.py
```

Resultado:

```text
r=10.277159 S=1.000000000000e+00
r=11.700427 S=9.907697363542e-01
r=15.000000 S=5.000000000000e-01
r=19.663672 S=3.439470930289e-13
```

Auditoria radial W1 original despues del fix:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\audit_w1_shift_profile.py `
  --profile original `
  --v-warp 0.02 `
  --output-dir outputs\codex_w1_shift_profile_after_fix
```

Resultado clave:

| r | S_raw | S_smoothed | S_eff grid | g01 grid | dS/dr |
| --- | --- | --- | --- | --- | --- |
| 10.277159 | 1.000000e0 | 9.987902e-1 | 9.987902e-1 | -1.997580e-2 | -4.112755e-3 |
| 11.700427 | 9.907697e-1 | 9.677666e-1 | 9.677666e-1 | -1.935533e-2 | -5.079121e-2 |
| 15.000000 | 5.000000e-1 | 5.000000e-1 | 5.001287e-1 | -1.000257e-2 | -1.930217e-1 |
| 19.663672 | 3.440535e-13 | 1.473375e-3 | 1.473375e-3 | -2.946749e-5 | -4.810163e-3 |

Esto explica el diagnostico anterior: antes del fix, en `r~11.7` se veia
`g01~-9.07e-4`, equivalente a `S_eff~0.045`. Despues del fix, el mismo radio da
`g01~-1.94e-2`, equivalente a `S_eff~0.968`, que es lo esperado para una meseta
interior suavizada.

Rerun de tabla de evidencia despues del fix:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\fuchs_w1_final_evidence_table.py `
  --profile original `
  --velocities 0,0.004,0.005,0.01,0.02 `
  --num-vecs 4 `
  --num-time-vecs 5 `
  --output-dir outputs\codex_final_evidence_after_swarp_fix_n4
```

Resultado:

| vWarp | NEC min | NEC frac | DEC min | max \|T01\| | min T00 | Tkk | diag | offdiag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.822399e39 | 0 | 3.822713e39 | 0 | 5.872160e39 | 1.822399e39 | 2.030613e39 | -2.082137e38 |
| 0.004 | 1.824828e39 | 0 | 3.825080e39 | 7.313834e38 | 5.872160e39 | 1.824828e39 | 2.030613e39 | -2.057848e38 |
| 0.005 | 1.825436e39 | 0 | 3.825655e39 | 9.142250e38 | 5.872160e39 | 1.825436e39 | 2.030613e39 | -2.051775e38 |
| 0.01 | 1.825821e39 | 0 | 3.811774e39 | 1.828379e39 | 5.872159e39 | 1.825821e39 | 1.887939e39 | -6.211842e37 |
| 0.02 | 1.792447e39 | 0 | 3.770787e39 | 3.656187e39 | 5.872159e39 | 1.792447e39 | 1.887940e39 | -9.549267e37 |

Rerun con 40 direcciones:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\fuchs_w1_final_evidence_table.py `
  --profile original `
  --velocities 0.02 `
  --num-vecs 40 `
  --num-time-vecs 10 `
  --output-dir outputs\codex_final_evidence_after_swarp_fix_n40
```

Resultado:

```text
vWarp=0.02
NEC min=1.365148e39, frac=0
WEC min=1.365148e39, frac=0
SEC min=9.757740e38, frac=0
DEC min=3.744716e39, frac=0
max |T01|=3.656187e39
min T00=5.872159e39
Tkk=1.365148e39
diag=4.509686e39
offdiag=-3.144538e39
```

Rerun paper-style con PNG:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\export_fuchs_w1_paper_regression.py `
  --profile original `
  --v-warp 0.02 `
  --num-vecs 40 `
  --num-time-vecs 10 `
  --critical-points 10 `
  --output-dir outputs\codex_paper_regression_after_swarp_fix_v002_n40 `
  --save-plots
```

Resultado:

```text
Material shell NEC min=1.3651e39
negative_fraction=0
```

Auditoria local del punto que antes era critico:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\audit_w1_critical_frame_tensor.py `
  --profile original `
  --v-warp 0.02 `
  --target 0.1,-11.7,0.0 `
  --num-vecs 40 `
  --output-dir outputs\codex_w1_critical_frame_tensor_after_swarp_fix
```

Resultado:

```text
coords=(0.1, -11.7, 0), r=11.7004
g01=-1.935533259556508e-02
T_hat00=1.3869683617553949e40
T_hat01=3.1784857182405853e39
Tkk pipeline=2.3345490694064312e39
Tkk manual=2.334549069406433e39
diag=5.4695e39
offdiag=-3.1349e39
```

Validaciones corridas despues del fix:

```text
tests/validate_compact_sigmoid_profile.py
tests/validate_fuchs_w1_shifted_after_swarp_fix.py
tests/validate_christoffel_physics.py
tests/validate_warp_shell_components.py
```

Todas pasan. Conclusion actualizada: el caso W1 original `vWarp=0.02` si se
reproduce como fisico en pyWarpFactory con el motor `christoffel`, una vez
corregida la orientacion de `Swarp`. El bug estaba antes de la curvatura, en la
forma radial efectiva del shift.

La regresion shifted protege tres puntos:

```text
g01(r~11.7004) = -1.935533259557e-02
NEC/WEC/SEC/DEC min > 0 en material_shell
Tkk(r~11.7004) = 2.3345e39 > 0
```

## Auditoria legacy: warpfactory_direct vs christoffel

Con `Swarp` corregido, se retomo `warpfactory_direct` como auditoria de motor
legacy, no como juez fisico. Se agrego:

- `Examples/compare_wfdirect_christoffel_w1.py`

Este comparador construye la misma metrica W1 post-fix y calcula dos cadenas de
curvatura completas:

```text
warpfactory_direct:
  get_ricci_tensor_warpfactory_direct -> RicciScalar -> EinsteinTensor
  -> EnergyTensor -> EnergyEulerianTensor -> mapas

christoffel:
  get_ricci_tensor -> RicciScalar -> EinsteinTensor
  -> EnergyTensor -> EnergyEulerianTensor -> mapas
```

Las compara por componente y region:

```text
passenger_core: r < 8
inner_transition: 9.5 < r < 12
middle_shell: 12 <= r < 18
outer_transition: 18 <= r < 20.5
exterior_vacuum: r > 22
material_shell: 10 <= r <= 20
```

Smoke test quick:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\compare_wfdirect_christoffel_w1.py `
  --profile quick `
  --case both `
  --output-dir outputs\codex_wfdirect_vs_christoffel_quick_smoke
```

Rerun original post-fix:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\compare_wfdirect_christoffel_w1.py `
  --profile original `
  --case both `
  --output-dir outputs\codex_wfdirect_vs_christoffel_original_after_swarp_fix
```

Artefactos:

- `outputs/codex_wfdirect_vs_christoffel_original_after_swarp_fix/wfdirect_vs_christoffel_w1.json`
- `outputs/codex_wfdirect_vs_christoffel_original_after_swarp_fix/wfdirect_vs_christoffel_w1.md`

Resultado original, caso estatico:

| field | component | region | abs_max | rel_max | r | direct | christoffel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RicciTensor | 33 | middle_shell | 3.773458e-03 | 1.549356e+00 | 15.527395 | -1.346448e-03 | 2.427009e-03 |
| RicciScalar | scalar | middle_shell | 7.546759e-03 | 1.617356e+00 | 15.527395 | -2.882024e-03 | 4.664734e-03 |
| EinsteinTensor | 11 | middle_shell | 3.033578e-03 | 1.478549e+00 | 14.501724 | 2.045980e-03 | -9.875978e-04 |
| EnergyTensor | 00 | middle_shell | 2.617432e40 | 9.426159e-01 | 14.682643 | 1.464665e39 | 2.763898e40 |
| EnergyEulerianTensor | 00 | middle_shell | 1.589435e40 | 9.498085e-01 | 15.180909 | 7.351958e38 | 1.662955e40 |

Condiciones en `material_shell`, estatico:

| engine | NEC min | NEC frac | WEC min | WEC frac | SEC min | DEC min | DEC frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| christoffel | 1.822399e39 | 0 | 1.822399e39 | 0 | 9.969510e38 | 3.822713e39 | 0 |
| warpfactory_direct | 6.358386e38 | 0 | -5.477418e36 | 0.024443 | 6.358386e38 | -6.726208e39 | 0.799973 |

Resultado original, caso shifted post-fix:

| field | component | region | abs_max | rel_max | r | direct | christoffel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RicciTensor | 33 | middle_shell | 3.770537e-03 | 1.547970e+00 | 15.540270 | -1.343684e-03 | 2.426853e-03 |
| RicciScalar | scalar | middle_shell | 7.540917e-03 | 1.613778e+00 | 15.540270 | -2.870397e-03 | 4.670520e-03 |
| EinsteinTensor | 22 | middle_shell | 3.028446e-03 | 1.480277e+00 | 14.501724 | 2.040831e-03 | -9.876152e-04 |
| EnergyTensor | 00 | middle_shell | 2.616957e40 | 9.423604e-01 | 14.682643 | 1.472580e39 | 2.764215e40 |
| EnergyEulerianTensor | 00 | middle_shell | 1.589435e40 | 9.494841e-01 | 15.180909 | 7.391002e38 | 1.663345e40 |

Condiciones en `material_shell`, shifted:

| engine | NEC min | NEC frac | WEC min | WEC frac | SEC min | DEC min | DEC frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| christoffel | 1.792447e39 | 0 | 1.792447e39 | 0 | 9.969537e38 | 3.770787e39 | 0 |
| warpfactory_direct | 1.511394e38 | 0 | -8.218469e38 | 0.221992 | 1.511394e38 | -8.320663e39 | 0.799973 |

Conclusion legacy:

```text
La primera divergencia esta en RicciTensor, antes de RicciScalar/Einstein/Energy/frame/mapas.
La discrepancia fuerte esta en middle_shell, no en passenger_core ni como problema
primario de mapas de condiciones.
warpfactory_direct introduce violaciones WEC/DEC falsas incluso en W1 estatico.
```

Por tanto, `solver_method="warpfactory_direct"` queda clasificado como:

```text
experimental / legacy-audit only
```

Para conclusiones fisicas usar:

```text
solver_method="christoffel"
```

Siguiente bisturi recomendado:

1. auditar `takeFiniteDifference1/2` con metricas manufacturadas no triviales;
2. instrumentar `get_ricci_tensor_warpfactory_direct` termino por termino;
3. comparar una version Christoffel que use exactamente los mismos `diff_1_gl`
   y `diff_2_gl` que `ricciT`.

### Primer bisturi legacy: derivadas manufacturadas

Se agrego `tests/validate_manufactured_finite_derivatives.py` para verificar las
derivadas finitas contra una funcion polinomial con derivadas mixtas no nulas:

```text
f = a + b*t + c*x + d*y^2 + e*z^3
    + h*x*y + q*y*z + p*x*z + u*t*x + v*t*y
```

La prueba cubre:

```text
dt, dx, dy, dz
dtt, dxx, dyy, dzz
dtx, dty, dtz, dxy, dxz, dyz
simetria de derivadas mixtas
```

Resultado local:

```text
errores interiores ~1e-14
Manufactured finite-derivative checks passed.
```

Esto reduce la sospecha sobre stencils/ejes/spacing basicos. El bug restante de
`warpfactory_direct` queda mas probablemente en la expansion directa `ricciT` o
en como sus terminos se contraen, no en `takeFiniteDifference1/2` para funciones
suaves manufacturadas.

Tambien se agrego una advertencia runtime en `solve_energy_tensor`:

```text
solver_method='warpfactory_direct' is experimental/legacy-audit only...
```

para evitar que se use accidentalmente como motor fisico.

### Segundo bisturi legacy: Christoffel con las mismas derivadas de `ricciT`

Se agrego una prueba de separacion entre:

```text
A = christoffel actual
B = Christoffel reconstruido usando exactamente diff_1_gl/diff_2_gl de warpfactory_direct
C = warpfactory_direct / expansion directa tipo ricciT
```

Archivos:

- `warpfactory/solver/tensor_utils.py::get_warpfactory_direct_metric_derivatives`
- `warpfactory/solver/tensor_utils.py::get_christoffel_from_metric_derivatives`
- `warpfactory/solver/tensor_utils.py::get_christoffel_derivative_from_metric_derivatives`
- `warpfactory/solver/tensor_utils.py::get_ricci_tensor_christoffel_from_warpfactory_derivatives`
- `Examples/audit_wfdirect_ricci_bridge.py`
- `tests/validate_warpfactory_direct_ricci_bridge.py`

Prueba local:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  tests\validate_warpfactory_direct_ricci_bridge.py
```

Resultado:

```text
WarpFactory direct vs same-derivative Christoffel bridge max error = 1.8575e-16
WarpFactory direct Ricci bridge equivalence check passed.
```

Rerun W1 original:

```powershell
C:\Users\Nelson\anaconda3\envs\astra\python.exe `
  Examples\audit_wfdirect_ricci_bridge.py `
  --profile original `
  --case both `
  --output-dir outputs\codex_wfdirect_ricci_bridge_original
```

Resultado estatico:

| comparison | component | region | abs_max | rel_max | r | candidate | reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bridge_vs_current | 33 | middle_shell | 3.773458e-03 | 1.549356e+00 | 15.527395 | -1.346448e-03 | 2.427009e-03 |
| direct_vs_current | 33 | middle_shell | 3.773458e-03 | 1.549356e+00 | 15.527395 | -1.346448e-03 | 2.427009e-03 |
| direct_vs_bridge | 12 | exterior_vacuum | 5.287329e-15 | 4.771828e-12 | 24.288680 | -9.443503e-04 | -9.443503e-04 |

En el peor punto estatico:

```text
idx=[0,145,72,1], coords=(-0.9,-15.5,-0.2), r=15.527395
R33 christoffel actual        =  2.427009e-03
R33 same-deriv Christoffel    = -1.346448e-03
R33 warpfactory_direct        = -1.346448e-03
direct - bridge               =  2.168404e-19
```

Resultado shifted post-fix:

| comparison | component | region | abs_max | rel_max | r | candidate | reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bridge_vs_current | 33 | middle_shell | 3.770537e-03 | 1.547970e+00 | 15.540270 | -1.343684e-03 | 2.426853e-03 |
| direct_vs_current | 33 | middle_shell | 3.770537e-03 | 1.547970e+00 | 15.540270 | -1.343684e-03 | 2.426853e-03 |
| direct_vs_bridge | 12 | exterior_vacuum | 5.287329e-15 | 4.771828e-12 | 24.288680 | -9.443503e-04 | -9.443503e-04 |

Conclusion actualizada del bisturi:

```text
warpfactory_direct C coincide con el Christoffel B construido a partir de sus
propios diff_1_gl/diff_2_gl. Por tanto, la expansion algebraica directa no parece
tener un bug grueso de indices/signos: reproduce la geometria que implican esas
derivadas discretas.

La discrepancia no es C vs B, sino B/C vs A. Es decir, el problema operativo es
la discretizacion de curvatura basada en segundas derivadas directas de la
metrica frente a la ruta que deriva los Christoffel discretos.
```

Esto corrige el ranking de sospechosos: `takeFiniteDifference1/2` pasa funciones
manufacturadas, pero la composicion `diff_1_gl/diff_2_gl -> Ricci` de la ruta
legacy no reproduce el motor fisico `christoffel` en W1. Si se quiere "arreglar"
`warpfactory_direct` para uso fisico dentro de pyWarpFactory, el arreglo practico
no es retocar un signo aislado en `ricciT`, sino reemplazar o delegar esa ruta a
la evaluacion Christoffel recomendada. Si se quiere mantener compatibilidad
historica con MATLAB, entonces debe seguir marcada como `legacy-audit only`.
