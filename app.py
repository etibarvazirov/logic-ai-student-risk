import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

from train_and_eval import run_full_pipeline

# -------------------------------------------------
# Səhifə parametrləri
# -------------------------------------------------
st.set_page_config(
    page_title="Tələbə Uğur Proqnozu (LTN)",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Tələbə Uğur Proqnozu")
st.write("""
Bu tətbiqin məqsədi risk altında olan şagirdləri (yəni imtahandan keçməmək ehtimalı yüksək olanları)
daha əvvəlcədən müəyyənləşdirməkdir. 
Biz iki modeli müqayisə edirik:
- Sadə Neyron Şəbəkə (Standart NN)
- Məntiq Qaydaları ilə Zənginləşdirilmiş Model (LTN modeli)

LTN modeli yalnız rəqəmlərə baxmır. O, müəllimin pedaqoji biliklərini
(qiymətlərin səviyyəsi, dərsə davamiyyət, əvvəlki uğursuzluqlar və s.) "qayda" kimi nəzərə alır.
""")

# -------------------------------------------------
# Nəticələri yükləyək və keşləyək (yenidən tren eləməsin hər dəyişiklikdə)
# -------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_results():
    return run_full_pipeline(csv_path="student-mat.csv")

results = get_results()

# İki modelin nəticələrini daha rahat istifadə üçün çıxaraq
std_results = {
    "label": "Standart Neyron Şəbəkə",
    "accuracy": results["std"]["accuracy"],
    "f1_fail": results["std"]["f1_fail"],
    "threshold": results["std"]["threshold"],
    "cm": np.array(results["std"]["cm"]),
    "rule_penalty": results["std"]["rule_penalty"],
}

ltn_results = {
    "label": "LTN modeli (Neyron Şəbəkə + Məntiq Qaydaları)",
    "accuracy": results["ltn"]["accuracy"],
    "f1_fail": results["ltn"]["f1_fail"],
    "threshold": results["ltn"]["threshold"],
    "cm": np.array(results["ltn"]["cm"]),
    "rule_penalty": results["ltn"]["rule_penalty"],
}

# -------------------------------------------------
# KÖMƏKÇİ FUNKSİYALAR (qrafiklər üçün)
# -------------------------------------------------
def plot_confusion_matrix(cm, title='Qarışıqlıq Matrisi'):
    """
    cm formatı:
    [[TN, FP],
     [FN, TP]]
    0 = Uğursuz (keçmir), 1 = Uğurlu (keçir)
    """
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['Proqnoz: Uğursuz', 'Proqnoz: Uğurlu'])
    ax.set_yticklabels(['Həqiqi: Uğursuz', 'Həqiqi: Uğurlu'])
    ax.set_title(title)

    # rəqəmləri göstər
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i,j],
                ha='center', va='center',
                color='black', fontsize=11, fontweight='bold'
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_performance_bars(std_results, ltn_results):
    """
    Accuracy və "Uğursuz sinfi" üçün F1 müqayisəsi (yəni riskli şagirdi tapmaq bacarığı).
    """
    etiketler = ['Dəqiqlik (Accuracy)', 'Risk Qrupunu Tapma (F1 - Uğursuz)']
    std_vals = [std_results["accuracy"], std_results["f1_fail"]]
    ltn_vals = [ltn_results["accuracy"], ltn_results["f1_fail"]]

    x = np.arange(len(etiketler))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(x - width/2, std_vals, width, label=std_results["label"])
    ax.bar(x + width/2, ltn_vals, width, label=ltn_results["label"])

    ax.set_ylim(0,1)
    ax.set_ylabel('Göstərici (0-1)')
    ax.set_title('Modellərin Müqayisəsi')
    ax.set_xticks(x)
    ax.set_xticklabels(etiketler, rotation=10)
    ax.legend(loc='lower right')

    # barların üstündə rəqəmlər
    for i,v in enumerate(std_vals):
        ax.text(x[i]-width/2, v+0.01, f"{v:.2f}",
                ha='center', fontsize=9)
    for i,v in enumerate(ltn_vals):
        ax.text(x[i]+width/2, v+0.01, f"{v:.2f}",
                ha='center', fontsize=9)

    fig.tight_layout()
    return fig


def plot_rule_penalty(std_results, ltn_results):
    """
    Rule Penalty = Qayda Pozuntusu.
    Aşağı olduqca yaxşıdır (müəllimin qoyduğu məntiqi qaydalar daha çox qorunur).
    """
    labels = [std_results["label"], ltn_results["label"]]
    vals = [std_results["rule_penalty"], ltn_results["rule_penalty"]]

    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(labels, vals)
    ax.set_ylabel('Qayda Pozuntusu (aşağı = daha yaxşı)')
    ax.set_title('Məntiq Qaydalarına Uyğunluq')

    for i,v in enumerate(vals):
        offset = v * 0.2 if v > 0 else 0.00005
        ax.text(
            i, v + offset,
            f"{v:.6f}",
            ha='center', fontsize=9
        )
    fig.tight_layout()
    return fig

# -------------------------------------------------
# 1. Şagird üçün fərdi proqnoz bloku (İndi ən birinci gəlir)
# -------------------------------------------------

st.header("1. Şagird üçün fərdi analiz / proqnoz")

st.write("""
Aşağıdakı slidlər vasitəsilə şagird haqqında məlumat daxil edin.
Sonra "Proqnoz et" düyməsini sıxın.
Sistem bu profili necə qiymətləndirəcəyini və risk dərəcəsini izah edəcək.
""")

col_input_left, col_input_right = st.columns(2)

with col_input_left:
    age = st.slider("Yaş", 15, 22, 17)
    G1 = st.slider("G1 (1-ci dövr balı, 0-20)", 0, 20, 10)
    G2 = st.slider("G2 (2-ci dövr balı, 0-20)", 0, 20, 10)
    studytime = st.slider("Oxuma vaxtı (1 = az ... 4 = çox)", 1, 4, 2)

with col_input_right:
    failures = st.slider("Keçmişdə neçə dəfə kəsilib?", 0, 4, 1)
    absences = st.slider("Buraxılmış dərs sayı", 0, 50, 5)
    # Burada əlavə ola bilərdik: "romantic münasibət var? internet var? və s."
    # Amma bunlar daha səs-küylüdür, əsas risk faktorlarını saxlayırıq.

# İstifadəçi düyməni sıxanda analiz et
if st.button("🔍 Proqnoz et"):
    risk_msgs = []
    protect_msgs = []

    # Bu hissə bizim qaydaların sadələşdirilmiş versiyasıdır:
    # Qayda A: Çox zəif ballar + çox kəsilmə + çox dərs buraxma = yüksək risk
    if (G1 < 8 and G2 < 8 and failures >= 2 and absences >= 15):
        risk_msgs.append(
            "Şagirdin G1 və G2 balları aşağıdır, çoxsaylı kəsilmə var və dərs buraxmaları çoxdur. "
            "Bu profil yüksək risk daşıyır (imtahandan keçməmə ehtimalı yüksəkdir)."
        )

    # Qayda B: Yüksək bal + kəsilmə yoxdur + ciddi oxuma = təhlükə azdır
    if (G1 >= 15 and G2 >= 15 and failures == 0 and studytime >= 3):
        protect_msgs.append(
            "Şagirdin balları yüksəkdir, kəsilməsi yoxdur və kifayət qədər oxuyur. "
            "Keçmə ehtimalı çox yüksəkdir."
        )

    # Heç biri yoxdursa - orta zona
    if len(risk_msgs) == 0 and len(protect_msgs) == 0:
        st.info(
            "Bu şagird nə tam təhlükəli zona, nə də tam təhlükəsiz zonadadır. "
            "Şagirdin vəziyyətini diqqətlə izləmək məsləhətdir."
        )
    else:
        for msg in risk_msgs:
            st.error(msg)
        for msg in protect_msgs:
            st.success(msg)

    st.caption("""
Bu izah LTN modelində istifadə etdiyimiz pedaqoji qaydaların sadələşdirilmiş formasıdır.
Real model bu qaydaları itirmədən neyron şəbəkə ilə birləşdirir.
""")


# -------------------------------------------------
# 2. Modellərin ümumi nəticələri (Train/Test nəticələri)
#    İstəyə bağlı göstərilsin (checkbox ilə)
# -------------------------------------------------

st.header("2. Modellərin ümumi nəticələri (dərsin bütün məlumatı üzrə)")

st.write("""
Aşağıdakı hissə bütün dataset üzərindən əldə olunan yekun performansdır.
Burada hər iki modelin nə qədər yaxşı işlədiyi göstərilir.
""")

show_metrics = st.checkbox("Ümumi göstəriciləri göstər (Dəqiqlik, F1, Qayda pozuntusu və s.)")

if show_metrics:
    colA, colB = st.columns(2)

    with colA:
        st.subheader(std_results["label"])
        st.metric("Dəqiqlik (Accuracy)", f"{std_results['accuracy']:.3f}")
        st.metric("Risk Qrupunu Tapma (F1 - Uğursuz)", f"{std_results['f1_fail']:.3f}")
        st.metric("Qərar həddi (threshold)", f"{std_results['threshold']:.3f}")
        st.metric("Qayda pozuntusu (Rule Penalty ↓ yaxşıdır)", f"{std_results['rule_penalty']:.6f}")

    with colB:
        st.subheader(ltn_results["label"])
        st.metric("Dəqiqlik (Accuracy)", f"{ltn_results['accuracy']:.3f}")
        st.metric("Risk Qrupunu Tapma (F1 - Uğursuz)", f"{ltn_results['f1_fail']:.3f}")
        st.metric("Qərar həddi (threshold)", f"{ltn_results['threshold']:.3f}")
        st.metric("Qayda pozuntusu (Rule Penalty ↓ yaxşıdır)", f"{ltn_results['rule_penalty']:.6f}")

    st.write("""
    Nə üçün bu vacibdir?

    - **Dəqiqlik (Accuracy):** Ümumilikdə neçə tələbəni düzgün proqnoz edirik.
    - **F1 (Uğursuz sinfi):** Zəif nəticə riski olan tələbələri nə qədər düzgün tapırıq.
    - **Qayda pozuntusu:** Model müəllimin məntiqinə nə qədər hörmət edir.
      Aşağı olduqca yaxşıdır, çünki səhv "təhlükəsizdir" demir və ya əksinə əsassız panika yaratmır.
    """)


# -------------------------------------------------
# 3. Vizual müqayisə (Confusion Matrix, Bar Chart və s.)
#    Bunları da checkbox ilə açırıq
# -------------------------------------------------

st.header("3. Vizual müqayisələr")

show_plots = st.checkbox("Qrafikləri göstər (Qarışıqlıq Matrisi, Müqayisə Bar Chart)")

if show_plots:
    st.subheader("Qarışıqlıq Matrisi (Confusion Matrix)")

    colCM1, colCM2 = st.columns(2)
    with colCM1:
        st.markdown(f"**{std_results['label']}**")
        fig_std_cm = plot_confusion_matrix(std_results["cm"], "Standart model")
        st.pyplot(fig_std_cm)

    with colCM2:
        st.markdown(f"**{ltn_results['label']}**")
        fig_ltn_cm = plot_confusion_matrix(ltn_results["cm"], "LTN modeli")
        st.pyplot(fig_ltn_cm)

    st.write("""
    İzah:
    - "Həqiqi: Uğursuz / Proqnoz: Uğurlu" (yəni riskli tələbəni qaçırmaq) bizim üçün ən təhlükəli hadisədir.
    - "Həqiqi: Uğurlu / Proqnoz: Uğursuz" isə boşuna panika yaratmaqdır.
    LTN modeli bu balansı daha etik formada qorumağa çalışır.
    """)

    st.subheader("Performans Müqayisəsi")

    fig_perf = plot_performance_bars(std_results, ltn_results)
    st.pyplot(fig_perf)

    st.subheader("Qaydalara Uyğunluq")

    fig_rule = plot_rule_penalty(std_results, ltn_results)
    st.pyplot(fig_rule)

    st.caption("""
    Burada görünür ki, LTN modeli həm daha yaxşı göstəricilər verir,
    həm də qaydaları daha az pozur. Bu o deməkdir ki
    LTN müəllim məntiqini (davamiyyət, əvvəlki nəticələr və s.) nəzərə alaraq qərar verir.
    """)

# -------------------------------------------------
# Alt izah / müəllif
# -------------------------------------------------
st.markdown("---")
st.caption("""
Bu sistem tələbə performansını proqnozlaşdırmaq üçün Neyron Şəbəkələri
və Məntiqi Qaydaları birləşdirir (LTN yanaşması).
Məqsəd yalnız balları təxmin etmək deyil, eyni zamanda
risk altında olan tələbəni müəllimə vaxtında göstərməkdir.
""")
