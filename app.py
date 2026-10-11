"""A guided classroom lesson built entirely from a constructed fictional case."""

import json

import pandas as pd
import streamlit as st

from court_assignment_audit.binomial_view import build_binomial_view
from court_assignment_audit.lesson import (
    N_CASES,
    OBSERVED_A,
    exact_count_probability,
    exact_tail_probability,
    lesson_csv,
    lesson_manifest,
    lesson_records,
    simulate_batch,
    target_result,
)
from court_assignment_audit.lesson_charts import (
    build_lesson_distribution,
    build_simulation_chart,
)

st.set_page_config(
    page_title="¿Diez veces en el mismo juzgado?", page_icon="🎲", layout="centered"
)


def percent(value: float, digits: int | None = None) -> str:
    """Use a decimal comma without rounding a small practice probability to zero."""
    if digits is None:
        digits = 2 if value >= 0.001 else 4
    return f"{100 * value:.{digits}f}".replace(".", ",") + " %"


@st.cache_data(show_spinner=False)
def classroom_simulation(batch_number: int):
    return simulate_batch(batch_number)


st.html(
    """
<style>
  .stMainBlockContainer {max-width: 1040px; padding-top: 4.75rem; padding-bottom: 4rem;}
  h1 {max-width: 820px; letter-spacing: -.035em; line-height: 1.08 !important;}
  h2 {letter-spacing: -.02em; padding-top: .4rem !important;}
  p, li {line-height: 1.7;}
  .lesson-eyebrow {font-size: .76rem; font-weight: 700; letter-spacing: .12em;
    color: #526674; margin: 0 0 1rem; text-transform: uppercase;}
  .lesson-lead {font-size: 1.2rem; line-height: 1.7; max-width: 820px; margin-bottom: 1rem;}
  .lesson-nav {display: flex; flex-wrap: wrap; gap: .55rem 1.2rem;
    padding: .8rem 0 1.8rem; font-size: .9rem;}
  .lesson-nav a {color: #136F83; text-decoration: none; border-bottom: 1px solid #C6D9DE;}
  .lesson-nav a:focus-visible {outline: 3px solid #B55915; outline-offset: 5px;}
  .lesson-note {border-left: 3px solid #167D9A; padding: .3rem 0 .3rem 1rem;
    margin: 1rem 0 1.5rem; color: #304A59;}
  .case-grid {display: grid; grid-template-columns: repeat(10, minmax(0, 1fr));
    gap: 8px; padding: 0 !important; margin: 1rem 0 !important; list-style: none;}
  .case-grid li {list-style: none; border: 1px solid #CDD9DF; border-radius: 8px;
    background: #F0F5F6; padding: .35rem .15rem .45rem; text-align: center; color: #324C59;}
  .case-grid .case-a {background: #FFF0DD; color: #85420D; border-color: #C77732;}
  .case-id {display: block; font-size: .68rem; line-height: 1.5; color: #586C78;}
  .case-court {display: block; font-size: 1.18rem; font-weight: 700; line-height: 1.6;}
  .court-grid {display: grid; grid-template-columns: repeat(12, minmax(0, 1fr));
    gap: 7px; padding: 0 !important; margin: 1rem 0 !important; list-style: none;}
  .court-grid li {list-style: none; text-align: center; border: 1px solid #9DB6C0;
    border-radius: 6px; background: #F0F5F6; padding: .45rem 0; font-weight: 700;}
  .court-grid span {display: block; font-size: .7rem; font-weight: 400; color: #4F6672;}
  .lesson-result {border-top: 2px solid #C06C22; border-bottom: 1px solid #DFC9B3;
    background: #FFF7EC; padding: 1.4rem 1.5rem; margin: .7rem 0 1rem;}
  .lesson-result .result-number {display: block; font-size: clamp(1.8rem, 5vw, 2.8rem);
    letter-spacing: -.03em; font-weight: 750; color: #85420D; line-height: 1.2;}
  .lesson-result p {margin: .65rem 0 0;}
  .lesson-equation {font-size: 1.35rem; font-weight: 650; color: #85420D; padding: .3rem 0;}
  @media (max-width: 650px) {
    .stMainBlockContainer {padding-top: 4.75rem; padding-left: 1rem; padding-right: 1rem;}
    .case-grid {grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 6px;}
    .court-grid {grid-template-columns: repeat(6, minmax(0, 1fr));}
    .lesson-lead {font-size: 1.07rem;}
    .lesson-result {padding: 1.15rem;}
  }
</style>
"""
)

records = lesson_records()
observed_view = build_binomial_view(target_result())
exact_ten = exact_count_probability(OBSERVED_A)
tail_ten = exact_tail_probability(OBSERVED_A)

st.html('<p class="lesson-eyebrow">Una historia para aprender estadística · Datos inventados</p>')
st.title("¿Diez veces en el mismo juzgado?")
st.html(
    '<p class="lesson-lead">Una empresa ficticia tiene <strong>60 expedientes</strong>. '
    'Diez fueron asignados al <strong>juzgado A</strong>. Si había doce juzgados con '
    'las mismas chances, ¿qué tan común sería un resultado así?</p>'
)
st.caption(
    "Construimos este caso para aprender: no proviene de una empresa ni de expedientes reales. "
    "Elegimos estos números para la lección; no los obtuvimos de una muestra aleatoria."
)
st.html(
    '<nav class="lesson-nav" aria-label="Recorrido de la lección">'
    '<a href="#el-caso">El caso</a><a href="#la-regla">La regla</a>'
    '<a href="#repetir">Repetir</a><a href="#calcular">Calcular</a>'
    '<a href="#concluir">Concluir</a><a href="#probar">Probá vos</a></nav>'
)

st.header("1. Empecemos por los expedientes", anchor="el-caso")
st.write(
    "Cada ficha es un expediente de la **Empresa Ejemplo**, una empresa inventada. "
    "La letra indica el juzgado que le tocó. Fijate en las diez fichas del juzgado A."
)
case_cards = "".join(
    f'<li class="{"case-a" if record["court"] == "A" else "case-other"}" '
    f'aria-label="Expediente {index:02d}, juzgado {record["court"]}">'
    f'<span class="case-id">{index:02d}</span>'
    f'<span class="case-court">{record["court"]}</span></li>'
    for index, record in enumerate(records, start=1)
)
st.html(f'<ol class="case-grid" aria-label="Los 60 expedientes ficticios">{case_cards}</ol>')
st.caption(
    "10 expedientes en A · 50 en los demás juzgados · Cada expediente se cuenta una sola vez."
)
st.html(
    '<p class="lesson-note">Para esta lección, fijamos de antemano una sola pregunta: '
    '<strong>¿el juzgado A recibió un número inusualmente alto de expedientes?</strong> '
    'No elegimos el juzgado que más se destaca después de mirar los resultados.</p>'
)

st.header("2. Acordemos la regla del reparto", anchor="la-regla")
st.write(
    "Imaginá esta regla: para cada expediente se sortea uno de los **12 juzgados**, "
    "todos con la misma chance. En el siguiente sorteo vuelven a participar los doce."
)
court_tiles = "".join(
    f'<li aria-label="Juzgado {court}, probabilidad uno en doce">'
    f'{court}<span>1 de 12</span></li>'
    for court in "ABCDEFGHIJKL"
)
st.html(f'<ul class="court-grid" aria-label="Doce juzgados equiprobables">{court_tiles}</ul>')
st.write(
    "Si sólo miramos A, cada sorteo tiene dos resultados: **sí, fue a A** (1 de 12), "
    "o **no, fue a otro juzgado** (11 de 12). Lo que sale en un sorteo no cambia "
    "las chances del siguiente: "
    "eso es lo que acá llamamos *independencia*."
)
st.markdown("En 60 expedientes, el promedio para A sería **60 ÷ 12 = 5**.")
st.html(
    '<p class="lesson-note"><strong>Esperar 5 no obliga a obtener 5.</strong> '
    'Es el promedio de muchos repartos. En uno solo podrían salir 3, 6, 10 u otro número. '
    'No hay un cupo de cinco expedientes por juzgado.</p>'
)

st.header("3. Repitamos el reparto", anchor="repetir")
st.write(
    "Ahora dejemos el caso original a un lado por un momento. En cada repetición vamos a "
    "sortear **60 expedientes nuevos** con esa misma regla y contar cuántos terminan en A. "
    "Después vamos a repetir el experimento **1.000 veces**."
)
st.markdown("**Un reparto completo → 60 sorteos → un conteo para A.**")
if "simulation_batch_number" not in st.session_state:
    st.session_state.simulation_batch_number = 0
if st.button("Simulá 1.000 repartos", type="primary", key="simulate"):
    st.session_state.simulation_batch_number += 1

batch_number = st.session_state.simulation_batch_number
batch = classroom_simulation(batch_number) if batch_number else None
if batch is None:
    st.caption(
        "Antes de apretar el botón, pensá: ¿en cuántos de esos 1.000 repartos A recibirá "
        "10 expedientes o más? El caso original seguirá igual."
    )
else:
    st.markdown(
        f"En el **lote {batch.batch_number}**, **{batch.exceedances} de los 1.000 repartos** "
        f"dieron **10 expedientes o más** a A: **{percent(batch.frequency)}** de este lote."
    )
    st.altair_chart(
        build_simulation_chart(batch.target_counts),
        width="stretch",
        alt=(
            "Histograma de 1.000 repartos simulados de 60 expedientes. Cada barra indica "
            "cuántos repartos dieron ese conteo al juzgado A. Se destacan los conteos de 10 o más."
        ),
    )
    st.caption(
        f"Lote {batch.batch_number} · Semilla {batch.seed} · Se cuentan los 1.000 repartos, "
        "sin descartar ninguno. Cada clic genera otro lote identificable."
    )
    st.write(
        "Al repetir, la frecuencia puede cambiar. La simulación nos ayuda a ver la variación; "
        "no modifica los 60 expedientes del ejemplo ni la probabilidad que calcularemos abajo."
    )
    if batch.exceedances == 0:
        st.caption("No verlo en este lote no significa que sea imposible.")

st.divider()
st.header("4. De la simulación a la probabilidad", anchor="calcular")
st.write(
    "La simulación cuenta lo que pasó en un lote. También podemos calcular la probabilidad "
    "de **cada conteo posible**, desde 0 hasta 60, bajo la regla acordada. "
    "Esa colección de probabilidades se llama **distribución binomial**."
)
st.markdown(
    f"Primero mirá la barra de **exactamente 10**: su probabilidad es **{percent(exact_ten)}**. "
    "La altura de cada barra corresponde a un resultado exacto."
)
st.altair_chart(
    build_lesson_distribution(observed_view, mode="breakdown"),
    width="stretch",
    alt=(
        "Distribución binomial para 60 expedientes y probabilidad uno en doce. "
        "La barra de exactamente 10 está destacada, junto con las de 11 o más. "
        "Una línea discontinua marca el promedio 5 y otra sólida el conteo observado 10."
    ),
)
st.caption(
    "Las líneas marcan 5 esperados en promedio y 10 observados. Mostramos la zona donde "
    "se concentra la probabilidad; todos los cálculos incluyen los conteos hasta 60, "
    "también los que quedan fuera del gráfico. Las barras conservan sus probabilidades originales."
)
st.subheader("La pregunta incluye 10… y también más de 10")
st.write(
    "Queríamos saber si A recibió muchos expedientes. Si 10 nos llama la atención, "
    "11, 12 o más también. Por eso sumamos la barra de 10 y **todas las que están a su derecha**. "
    "A ese conjunto lo llamamos la *cola* de la distribución."
)
st.html(
    '<p class="lesson-equation">Exactamente 10: 1,57 % → 10 o más: 2,58 %</p>'
)
st.write(
    "Esa probabilidad de **10 o más**, calculada bajo la regla propuesta, "
    "es el **valor p** de nuestra pregunta sobre un exceso en A."
)

st.header("5. Volvamos al caso", anchor="concluir")
st.html(
    '<div class="lesson-result" role="note" aria-label="Conclusión del ejemplo">'
    '<span class="result-number">Aproximadamente 26 de cada 1.000</span>'
    '<p>Si repitiéramos muchos repartos de <strong>60 expedientes</strong> con la regla '
    'de uno entre doce, alrededor de 26 de cada 1.000 le darían '
    '<strong>10 o más al juzgado A</strong>.'
    '</p></div>'
)
st.markdown(
    "En nuestro caso construido, A recibió **10 en lugar de 5 esperados en promedio**: "
    "el doble. Es un resultado poco común bajo esa regla, aunque **puede ocurrir**. "
    "Por sí solo, el conteo no explica por qué ocurrió."
)
st.write(
    "El **2,58 % no es la probabilidad de fraude**, ni la probabilidad de que la regla sea "
    "verdadera o falsa. Y un resultado común tampoco demostraría que un proceso real "
    "fue aleatorio. Acá estamos aprendiendo con un ejemplo enteramente ficticio."
)

st.divider()
st.header("6. Ahora probá vos", anchor="probar")
st.write(
    "¿Y si A hubiera recibido otro número de expedientes? Mové el deslizador y mirá "
    "cómo cambia la probabilidad de obtener **ese número o más**."
)
practice_count = st.slider(
    "Cantidad hipotética de expedientes en A", 0, 15, OBSERVED_A, key="practice_count"
)
practice_tail = exact_tail_probability(practice_count)
st.markdown(
    f"**{practice_count} o más de 60 → {percent(practice_tail)}**, "
    "si cada expediente tiene una chance entre doce de ir a A."
)
st.caption(
    "Sólo cambia esta pregunta de práctica. El conjunto original sigue teniendo "
    "60 expedientes y 10 asignados a A; la simulación anterior también se conserva. "
    "El deslizador llega a 15, pero cada cola se calcula hasta 60."
)
st.altair_chart(
    build_lesson_distribution(
        build_binomial_view(target_result(practice_count)), mode="tail", hypothetical=True
    ),
    width="stretch",
    alt=(
        f"La misma distribución binomial, con la cola desde {practice_count} hasta 60. "
        f"El umbral {practice_count} es hipotético; el conjunto original conserva 10 casos en A."
    ),
)
if practice_tail < 0.001:
    st.caption(
        "La cola es tan chica que sus barras casi no se ven en esta escala. "
        "La probabilidad sigue incluyendo todos los conteos hasta 60."
    )
if practice_count == 0:
    st.write("Cero o más incluye todos los resultados posibles. Por eso la probabilidad es 100 %.")
elif practice_count == 5:
    st.write(
        "El umbral coincide con el promedio. La probabilidad de 5 o más es **56,72 %**, "
        "no necesariamente la mitad: estamos contando resultados enteros, incluido el 5."
    )
elif practice_count == OBSERVED_A:
    st.write("Volviste al conteo del ejemplo: 10 o más tiene una probabilidad de **2,58 %**.")
else:
    st.write(
        "Al pedir un conteo más alto, quedan menos resultados en la cola. "
        "La probabilidad baja, aunque la regla del sorteo sigue siendo la misma."
    )

st.divider()
with st.expander("Para seguir: la fórmula y los supuestos"):
    st.write(
        "Llamamos X a la cantidad de expedientes que llegan a A en un reparto de 60. "
        "Usamos una binomial porque hay 60 sorteos independientes, cada uno con dos "
        "resultados respecto de A, y una probabilidad constante de 1/12."
    )
    st.latex(r"X\sim\operatorname{Binomial}(60,\tfrac{1}{12})")
    st.latex(r"P(X=k)=\binom{60}{k}(\tfrac{1}{12})^k(\tfrac{11}{12})^{60-k}")
    st.latex(r"P(X\geq10)=\sum_{k=10}^{60}P(X=k)")
    st.write(
        f"Con más decimales: exactamente 10 = **{percent(exact_ten, 8)}**; "
        f"10 o más = **{percent(tail_ten, 8)}**. "
        "La frecuencia simulada es un conteo del lote, no reemplaza este cálculo exacto."
    )
    st.write(
        "El modelo supone que los doce juzgados son elegibles en cada sorteo, que todos "
        "tienen la misma chance y que un expediente no condiciona otro. Si hubiera cupos, "
        "conexidad entre expedientes, competencias distintas o cambios en las reglas, "
        "haría falta otro modelo. No estamos describiendo un sistema judicial real."
    )

with st.expander("Datos del ejemplo y cómo repetir la experiencia"):
    st.write(
        "El CSV contiene los mismos 60 expedientes construidos que ves arriba. "
        "El registro JSON distingue el caso original, el lote simulado —si lo corriste— "
        "y la pregunta hipotética del deslizador. No hay datos reales ni archivos subidos."
    )
    st.download_button(
        "Descargá los 60 expedientes (CSV)",
        lesson_csv(),
        file_name="constructed_lesson_assignments.csv",
        mime="text/csv",
        key="download_cases",
    )
    st.download_button(
        "Descargá el registro de la experiencia (JSON)",
        json.dumps(
            lesson_manifest(batch=batch, practice_threshold=practice_count),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        file_name="lesson_run_manifest.json",
        mime="application/json",
        key="download_manifest",
    )
    st.dataframe(
        pd.DataFrame(records)[["case_id", "court"]].rename(
            columns={"case_id": "Expediente", "court": "Juzgado"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Caso fijo: {N_CASES} expedientes · 12 juzgados · 10 en A. "
        "Los clics de simulación se conservan mientras esta sesión siga abierta. "
        "Al recargar la página, la lección vuelve a su inicio."
    )
    st.write(
        "El README en inglés explica cómo ejecutar la lección y reproducir un lote. "
        "Las notas técnicas incluyen el análisis del reparto completo y las comparaciones "
        "entre juzgados como extensiones: responden otras preguntas."
    )
    st.markdown(
        "[Código y guía de reproducción](https://github.com/axel2307/court-assignment-audit)"
        " · [Notas técnicas](https://github.com/axel2307/court-assignment-audit/blob/main/"
        "docs/methodology.md)"
    )

st.caption(
    "Court Assignment Audit · Una lección con datos 100 % ficticios y cálculos reproducibles."
)
