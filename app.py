"""
====================================================================
 Azubi-Voice-Report – Prototyp
====================================================================
Eine Streamlit-App, mit der Azubis ihr Berichtsheft per Sprache
führen können.

Ablauf:
1. Nutzer wählt seinen Ausbildungsberuf aus
2. Nutzer nimmt eine Sprachnachricht im Browser auf (st.audio_input)
3. Die Audiodatei wird an Groq (Whisper large-v3) geschickt -> Transkript
4. Das Transkript wird an Groq (Llama/Qwen) geschickt -> veredelter
   Berichtsheft-Eintrag (sachlich, Vergangenheitsform, Stichpunkte)
5. Rohes Transkript + fertiger Bericht werden angezeigt
6. Der fertige Bericht kann in einer Historie gespeichert werden
   (session_state + optionale Textdatei)
7. Über "🔄 Neue Aufnahme starten" kann jederzeit neu aufgenommen
   werden – vorher wird IMMER mit Ja/Nein nachgefragt, da der
   aktuelle Rohtext + Bericht dabei gelöscht werden.
====================================================================
"""

import streamlit as st
from groq import Groq
from datetime import datetime
import os
import json

# --------------------------------------------------------------
# Grundkonfiguration der Streamlit-Seite
# --------------------------------------------------------------
st.set_page_config(
    page_title="Azubi-Voice-Report",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------------------
# Konstanten: Verwendete Modelle bei Groq
# --------------------------------------------------------------
WHISPER_MODEL = "whisper-large-v3"
LLAMA_MODEL = "qwen/qwen3.8-27b"

# Lokale Dateien zur persistenten Speicherung der Historie
HISTORY_FILE = "berichtsheft_historie.txt"
HISTORY_JSON_FILE = "berichtsheft_historie.json"


# --------------------------------------------------------------
# Liste der auswählbaren Ausbildungsberufe (inkl. Emoji fürs Auge)
# --------------------------------------------------------------
# Diese Liste kann beliebig erweitert werden. Der Beruf wird später
# an die KI übergeben, damit sie passende Fachbegriffe verwendet
# (z.B. "Instandsetzung einer Bremsanlage" statt "Auto repariert").
AUSBILDUNGSBERUFE = (
    "🔧 KFZ-Mechatroniker/-in",
    "💼 Bürokaufmann/-frau",
    "💻 Fachinformatiker/-in",
    "⚡ Elektroniker/-in",
    "🏭 Industriemechaniker/-in",
    "🍽️ Koch/Köchin",
    "🏥 Pflegefachkraft",
    "🛠️ Anlagenmechaniker/-in",
    "📦 Kaufmann/-frau für Groß- und Außenhandelsmanagement",
    "🌳 Gärtner/-in",
    "🎨 Mediengestalter/-in",
    "🏗️ Maurer/-in",
)


def lade_historie() -> list:
    """Lädt gespeicherte Einträge aus der JSON-Datei, falls vorhanden."""
    if os.path.exists(HISTORY_JSON_FILE):
        try:
            with open(HISTORY_JSON_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# --------------------------------------------------------------
# Session-State initialisieren
# --------------------------------------------------------------
# Wir nutzen st.session_state, damit Daten zwischen den einzelnen
# "Reruns" von Streamlit erhalten bleiben (Streamlit führt bei jeder
# Interaktion das komplette Skript erneut aus!).
if "transkript" not in st.session_state:
    st.session_state["transkript"] = ""

if "bericht" not in st.session_state:
    st.session_state["bericht"] = ""

if "historie" not in st.session_state:
    st.session_state["historie"] = lade_historie()

# Zähler, über den wir dem audio_input-Widget bei Bedarf einen neuen
# "key" geben. Das ist der einzige Weg, eine bereits gemachte
# Aufnahme aus dem Widget selbst zu entfernen (Streamlit erlaubt kein
# direktes "Leeren" eines audio_input-Widgets).
if "audio_key_counter" not in st.session_state:
    st.session_state["audio_key_counter"] = 0

# Flag: Zeigt an, ob gerade auf eine Ja/Nein-Bestätigung gewartet wird,
# bevor der aktuelle Rohtext + Bericht gelöscht werden dürfen.
if "warte_auf_bestaetigung" not in st.session_state:
    st.session_state["warte_auf_bestaetigung"] = False


# --------------------------------------------------------------
# Hilfsfunktion: Groq-Client erzeugen
# --------------------------------------------------------------
def get_groq_client(api_key: str) -> Groq:
    """
    Erstellt und gibt einen Groq-Client zurück.
    Wir kapseln das in einer Funktion, damit wir bei Bedarf
    (z.B. neuer API-Key eingegeben) einfach neu instanziieren können.
    """
    return Groq(api_key=api_key)


# --------------------------------------------------------------
# Funktion 1: Audio transkribieren (Speech-to-Text via Whisper)
# --------------------------------------------------------------
def transkribiere_audio(client: Groq, audio_bytes: bytes) -> str:
    """
    Schickt die aufgenommene Audiodatei an die Groq-API (Whisper)
    und gibt das deutsche Transkript als String zurück.

    Parameter:
        client (Groq): Der initialisierte Groq-Client
        audio_bytes (bytes): Die Rohdaten der Audioaufnahme

    Rückgabe:
        str: Das transkribierte Text (bei Fehler: leerer String)
    """
    try:
        # Groq erwartet eine Datei-ähnliche Struktur: (Dateiname, Bytes)
        transkription = client.audio.transcriptions.create(
            file=("aufnahme.wav", audio_bytes),
            model=WHISPER_MODEL,
            language="de",          # Deutsch fest vorgeben für Präzision
            response_format="text", # Wir wollen direkt reinen Text zurück
            temperature=0.0         # Möglichst deterministisch/genau
        )
        # Je nach SDK-Version kommt entweder ein String oder ein Objekt
        # mit .text zurück -> wir behandeln beide Fälle sicherheitshalber
        if isinstance(transkription, str):
            return transkription.strip()
        return getattr(transkription, "text", "").strip()

    except Exception as fehler:
        # Fehler dem Nutzer verständlich anzeigen, App nicht abstürzen lassen
        st.error(f"❌ Fehler bei der Transkription (Whisper): {fehler}")
        return ""


# --------------------------------------------------------------
# Funktion 2: Transkript in professionellen Berichtsheft-Eintrag
#             umwandeln (Text-Veredelung via Llama/Qwen)
# --------------------------------------------------------------
def veredle_text_zu_bericht(client: Groq, transkript: str, beruf: str) -> str:
    """
    Nimmt das rohe Transkript und lässt es von der KI (via Groq)
    in einen professionellen, sachlichen Berichtsheft-Eintrag
    umformulieren.

    Parameter:
        client (Groq): Der initialisierte Groq-Client
        transkript (str): Das rohe, gesprochene Transkript
        beruf (str): Der gewählte Ausbildungsberuf, damit die KI
                     passende Fachbegriffe verwendet (z.B. "Instandsetzung
                     einer Bremsanlage" statt nur "Auto repariert")

    Rückgabe:
        str: Der veredelte Berichtsheft-Text (bei Fehler: leerer String)
    """

    # Emoji aus dem Beruf-String entfernen, bevor er an die KI geht
    # (die KI braucht nur den reinen Berufsnamen als Kontext)
    beruf_bereinigt = "".join(
        zeichen for zeichen in beruf if zeichen.isalnum() or zeichen in " /-.äöüÄÖÜß"
    ).strip()

    # System-Prompt: Legt die "Rolle" und die Schreibregeln der KI fest
    system_prompt = (
        "Du bist ein erfahrener Ausbilder. Deine Aufgabe ist es, die täglichen Notizen "
        "eines Azubis in einen formellen Ausbildungsnachweis umzuschreiben.\n\n"
        f"Der Azubi befindet sich in der Ausbildung zum/zur: {beruf_bereinigt}.\n\n"
        "REGELN:\n"
        "- Verwende ausschließlich Fachbegriffe aus diesem Ausbildungsberuf.\n"
        "- Schreibe in kurzen, prägnanten Stichpunkten.\n"
        "- Sätze müssen im Präteritum oder Perfekt stehen (z.B. 'Durchgeführt', 'Unterstützt bei...').\n"
        "- Entferne Füllwörter wie 'eigentlich', 'dann', 'halt'.\n"
        "- Formuliere Alltagssprache in Fachsprache um (z.B. statt 'Kabel gezogen' schreibe 'Verlegung von Leitungen').\n"
        "- Falls der Azubi Berufsschule erwähnt, liste die Lernfelder auf.\n"
        "- Erfinde KEINE Tätigkeiten hinzu, die nicht im Transkript erwähnt wurden.\n"
        "- Gib NUR den fertigen Berichtsheft-Text zurück, ohne einleitende Sätze."
        "\n\nBeispiel für den Stil:\n"
        "Notiz: 'Hab heute mit dem Chef die Regale im Lager sortiert und neue Ware eingebucht.'\n"
        "Bericht:\n"
        "- Durchführung einer Bestandsaufnahme im Lager.\n"
        "- Kontrolle und systemseitige Erfassung des Wareneingangs.\n"
        "- Optimierung der Lagerplatzbelegung."
    )

    try:
        antwort = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    "Hier ist das rohe, gesprochene Transkript eines Azubis "
                    "über seinen heutigen Arbeitstag. Erstelle daraus einen "
                    "professionellen Berichtsheft-Eintrag gemäß den "
                    "Systemregeln:\n\n"
                    f"'{transkript}'"
                )}
            ],
            temperature=0.3,   # Niedrig, für konsistente/sachliche Ausgaben
            max_tokens=1024,
        )
        return antwort.choices[0].message.content.strip()

    except Exception as fehler:
        st.error(f"❌ Fehler bei der Text-Veredelung: {fehler}")
        return ""


# --------------------------------------------------------------
# Funktion 3: Bericht in Historie speichern (Session, JSON + TXT)
# --------------------------------------------------------------
def speichere_in_historie(bericht_text: str, datum_str: str = None, stunden: float = 8.0, beruf: str = ""):
    """
    Speichert den fertigen Berichtsheft-Eintrag
    a) im session_state (für die aktuelle Sitzung)
    b) in der JSON-Datei (persistent über App-Neustarts hinweg)
    c) zusätzlich in einer lokalen Textdatei als lesbare Übersicht
    """
    if not datum_str:
        datum_str = datetime.now().strftime("%d.%m.%Y")

    zeitstempel = datetime.now().strftime("%d.%m.%Y %H:%M")
    eintrag = {
        "datum": datum_str,
        "stunden": stunden,
        "beruf": beruf,
        "text": bericht_text,
        "erstellt_am": zeitstempel
    }

    # a) In der Session-Liste ablegen (oben einfügen = neueste zuerst)
    st.session_state["historie"].insert(0, eintrag)

    # b) In JSON-Datei sichern
    try:
        with open(HISTORY_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state["historie"], f, ensure_ascii=False, indent=2)
    except Exception as fehler:
        st.warning(f"⚠️ Konnte Historie nicht in JSON speichern: {fehler}")

    # c) Zusätzlich in Textdatei schreiben
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as datei:
            beruf_info = f" ({beruf})" if beruf else ""
            datei.write(f"\n=== Eintrag vom {datum_str} ({stunden} Std.){beruf_info} ===\n")
            datei.write(bericht_text.strip())
            datei.write("\n")
    except Exception as fehler:
        st.warning(f"⚠️ Eintrag konnte nicht zusätzlich in Datei "
                   f"gespeichert werden: {fehler}")


# --------------------------------------------------------------
# Funktion 4: Aktuellen Rohtext + Bericht zurücksetzen und die
#             Audio-Aufnahme fürs Widget "leeren"
# --------------------------------------------------------------
def setze_aufnahme_zurueck():
    """
    Löscht Transkript und Bericht aus dem session_state und erhöht
    den audio_key_counter, damit st.audio_input mit einem neuen
    (leeren) Widget neu gerendert wird.
    """
    st.session_state["transkript"] = ""
    st.session_state["bericht"] = ""
    st.session_state["audio_key_counter"] += 1


# ================================================================
#                         STREAMLIT UI
# ================================================================

st.title("🎙️ Azubi-Voice-Report")
st.caption("Berichtsheft per Sprache – einfach aufnehmen, KI erledigt den Rest.")

# --------------------------------------------------------------
# API-Key Eingabe (sicher, am Anfang der App)
# --------------------------------------------------------------
# Reihenfolge: 1) st.secrets (empfohlen, z.B. .streamlit/secrets.toml)
#              2) Umgebungsvariable
#              3) manuelles Eingabefeld als Fallback
# Wir greifen defensiv auf st.secrets zu, da ein Zugriff ohne
# vorhandene secrets.toml sonst zu einem Fehler führen kann.
try:
    secret_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    secret_key = ""

with st.sidebar:
    st.header("🔑 API-Konfiguration")

    if secret_key:
        api_key = secret_key
        st.success("✅ API-Key aus den Secrets geladen.")
    else:
        default_key = os.environ.get("GROQ_API_KEY", "")
        api_key = st.text_input(
            "Groq API-Key",
            value=default_key,
            type="password",
            help="Erhältlich unter https://console.groq.com/keys. "
                 "Der Key wird nur für diese Sitzung im Speicher gehalten."
        )

    if api_key:
        client = get_groq_client(api_key)

        with st.expander("🔍 Verfügbare Modelle prüfen"):
            try:
                models = client.models.list().data
                st.write("Dein Key unterstützt diese Modelle:")
                for m in models:
                    st.code(m.id)  # Zeigt die IDs zum Kopieren an
            except Exception as e:
                st.write(f"Modell-Liste konnte nicht geladen werden: {e}")

    st.divider()
    with st.expander("⚙️ Verwendete KI-Modelle"):
        st.markdown(
            f"- 🎧 **Transkription:** `{WHISPER_MODEL}`\n"
            f"- ✍️ **Text-Veredelung:** `{LLAMA_MODEL}`"
        )

# Ohne API-Key kann die App nicht sinnvoll weiterarbeiten
if not api_key:
    st.info("👈 Bitte gib links in der Seitenleiste deinen Groq API-Key ein, "
            "um die App zu nutzen.")
    st.stop()  # Beendet die Skript-Ausführung an dieser Stelle sauber

st.divider()

# --------------------------------------------------------------
# 🧑‍🔧 Berufsauswahl
# --------------------------------------------------------------
st.markdown("### 🧑‍🔧 Dein Ausbildungsberuf")

gewaehlter_beruf = st.selectbox(
    "Beruf auswählen",
    options=AUSBILDUNGSBERUFE,
    help="Die KI passt die Fachbegriffe im Bericht an diesen "
         "Ausbildungsberuf an."
)

st.info(f"🧑‍🔧 Aktiver Ausbildungsberuf: **{gewaehlter_beruf}**")
st.divider()

# --------------------------------------------------------------
# 🔒 Bestätigungsdialog für eine neue Aufnahme
# --------------------------------------------------------------
# Sobald "warte_auf_bestaetigung" True ist, wird NUR dieser Dialog
# angezeigt (Rest der App pausiert via st.stop()). So kann der
# Nutzer nicht versehentlich weiterklicken, ohne sich zu entscheiden.
if st.session_state["warte_auf_bestaetigung"]:
    st.warning(
        "⚠️ **Bist du sicher?** Wenn du eine neue Aufnahme startest, "
        "werden das aktuelle Transkript und der fertige Bericht "
        "gelöscht (sofern sie nicht bereits in der Historie "
        "gespeichert wurden)."
    )

    spalte_ja, spalte_nein = st.columns(2)

    with spalte_ja:
        if st.button("✅ Ja, löschen und neu aufnehmen",
                      type="primary", use_container_width=True):
            setze_aufnahme_zurueck()
            st.session_state["warte_auf_bestaetigung"] = False
            st.rerun()

    with spalte_nein:
        if st.button("❌ Nein, abbrechen", use_container_width=True):
            st.session_state["warte_auf_bestaetigung"] = False
            st.rerun()

    st.stop()  # Restliche UI (Rekorder, alter Bericht, ...) ausblenden

# --------------------------------------------------------------
# Schritt 1: Audio-Aufnahme
# --------------------------------------------------------------
st.subheader("1️⃣ Sprachnachricht aufnehmen")

kopf_links, kopf_rechts = st.columns([3, 1])
with kopf_links:
    st.write("Erzähle kurz, was du heute gemacht hast – die KI macht daraus "
             "einen sauberen Berichtsheft-Eintrag.")
with kopf_rechts:
    # Der Reset-Button ist nur sinnvoll, wenn es bereits etwas zu
    # löschen gibt (Transkript oder Bericht vorhanden).
    if st.session_state["transkript"] or st.session_state["bericht"]:
        if st.button("🔄 Neue Aufnahme starten", use_container_width=True):
            st.session_state["warte_auf_bestaetigung"] = True
            st.rerun()

# st.audio_input erzeugt ein Mikrofon-Widget direkt im Browser.
# Der "key" enthält den audio_key_counter -> wird dieser erhöht,
# rendert Streamlit ein komplett neues, leeres Widget.
audio_datei = st.audio_input(
    "Aufnahme starten",
    key=f"audio_input_{st.session_state['audio_key_counter']}"
)

# --------------------------------------------------------------
# Schritt 2 & 3: Verarbeitung, sobald eine Aufnahme vorliegt
# --------------------------------------------------------------
if audio_datei is not None:

    # Button, um die Verarbeitung explizit auszulösen
    # (verhindert unnötige/teure API-Calls bei jedem Rerun)
    if st.button("🚀 Bericht erstellen", type="primary"):

        # Die Rohbytes der Aufnahme auslesen
        audio_bytes = audio_datei.read()

        # --- Transkription ---
        with st.spinner("🎧 Transkribiere Audio mit Whisper..."):
            transkript = transkribiere_audio(client, audio_bytes)

        if transkript:
            st.session_state["transkript"] = transkript

            # --- Veredelung (Beruf wird für passende Fachbegriffe übergeben) ---
            with st.spinner("✍️ Erstelle professionellen Berichtseintrag..."):
                bericht = veredle_text_zu_bericht(
                    client, transkript, gewaehlter_beruf
                )

            if bericht:
                st.session_state["bericht"] = bericht
            else:
                # Bei Fehler in der Veredelung: alten Bericht zurücksetzen,
                # damit nicht ein veralteter Bericht zum neuen Transkript
                # passend angezeigt wird
                st.session_state["bericht"] = ""
        else:
            # Kein Transkript -> auch keinen alten Bericht mehr anzeigen
            st.session_state["transkript"] = ""
            st.session_state["bericht"] = ""

# --------------------------------------------------------------
# Schritt 4: Anzeige von Transkript und fertigem Bericht
# --------------------------------------------------------------
if st.session_state["transkript"]:
    st.divider()
    st.subheader("2️⃣ Rohes Transkript")
    st.text_area(
        "Was gesagt wurde:",
        value=st.session_state["transkript"],
        height=100,
        disabled=True
    )

if st.session_state["bericht"]:
    st.subheader("3️⃣ Fertiger Berichtsheft-Eintrag")

    col_datum, col_stunden = st.columns(2)
    with col_datum:
        eintrags_datum = st.date_input("📅 Datum des Tages:", value=datetime.today())
    with col_stunden:
        eintrags_stunden = st.number_input(
            "⏱️ Arbeitszeit (Std.):",
            min_value=0.5,
            max_value=24.0,
            value=8.0,
            step=0.5
        )

    bearbeiteter_bericht = st.text_area(
        "✏️ Berichtsheft-Text (kann vor dem Speichern angepasst werden):",
        value=st.session_state["bericht"],
        height=180,
        help="Du kannst den Text hier direkt bearbeiten, Fehler korrigieren oder Details ergänzen."
    )

    # --------------------------------------------------------------
    # Schritt 5: Speichern & Download
    # --------------------------------------------------------------
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 In Historie speichern", type="primary", use_container_width=True):
            datum_str = eintrags_datum.strftime("%d.%m.%Y")
            speichere_in_historie(
                bericht_text=bearbeiteter_bericht,
                datum_str=datum_str,
                stunden=eintrags_stunden,
                beruf=gewaehlter_beruf
            )
            st.success("✅ Eintrag wurde in der Historie gespeichert!")

    with col2:
        dateiname = f"berichtsheft_{eintrags_datum.strftime('%Y%m%d')}.txt"
        export_text = (
            f"Datum: {eintrags_datum.strftime('%d.%m.%Y')} ({eintrags_stunden} Std.)\n"
            f"Beruf: {gewaehlter_beruf}\n\n"
            f"{bearbeiteter_bericht.strip()}\n"
        )
        st.download_button(
            label="⬇️ Als Textdatei herunterladen",
            data=export_text,
            file_name=dateiname,
            mime="text/plain",
            use_container_width=True
        )

# --------------------------------------------------------------
# Historie anzeigen (in einem ausklappbaren Bereich)
# --------------------------------------------------------------
if st.session_state["historie"]:
    st.divider()
    kopf_hist_l, kopf_hist_r = st.columns([2, 1])
    with kopf_hist_l:
        st.subheader("📚 Bisherige Einträge (Historie)")
    with kopf_hist_r:
        alle_texte = []
        for e in st.session_state["historie"]:
            datum = e.get("datum", "Ohne Datum")
            std = f" ({e['stunden']} Std.)" if "stunden" in e else ""
            beruf_info = f" - {e['beruf']}" if e.get("beruf") else ""
            alle_texte.append(f"=== {datum}{std}{beruf_info} ===\n{e.get('text', '')}")

        st.download_button(
            label="📦 Alle als .txt",
            data="\n\n".join(alle_texte),
            file_name=f"berichtsheft_alle_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    for eintrag in st.session_state["historie"]:
        titel = f"📅 {eintrag.get('datum', 'Eintrag')}"
        if "stunden" in eintrag:
            titel += f" · ⏱️ {eintrag['stunden']} Std."
        if eintrag.get("beruf"):
            titel += f" · {eintrag['beruf']}"
        with st.expander(titel):
            st.markdown(eintrag.get("text", ""))