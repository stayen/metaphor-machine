# Metaphor Machine suite
#   - script to generate metaphor and metaphor chains for Suno/Producer.ai generative services
#
# A technique to generate style-only definitions using modular, poetic components
#
# Some Rights Reserved 2025
#
# This work is distributed under CC BY 4.0 (Attribution 4.0 International Deed)
# The full description: https://creativecommons.org/licenses/by/4.0/
#
# When using and/or distributing, add reference to
#
# Konstantin Boyandin <developer@boyandin.com>
# Site: https://metaphor-machine.com (the link may vary, please contact email above when necessary)
#

import re
import json
import random
import argparse
from typing import Mapping, Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml  # pip install pyyaml

# Global config holders (populated via CLI or programmatic calls)
STYLE_CONFIG: Dict[str, Any] = {}
PERSONAS_CONFIG: Dict[str, Any] = {}

# DATA CLASSES AND HELPERS
# ============================================================================ 

@dataclass
class StyleConfig:
    genre_eras: List[str]
    genre_subgenres: Dict[str, List[str]]
    intensity_adjectives: Dict[str, List[str]]
    delivery_nouns: Dict[str, List[str]]
    motion_verbs: List[str]
    musical_objects: Dict[str, List[str]]
    environments: List[str]
    sensory_mediums: Dict[str, List[str]]
    emotions: Dict[str, List[str]]
    emotional_arcs: Dict[str, List[str]]

def load_style_config(path: Path) -> StyleConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    return StyleConfig(
        genre_eras=raw["genre"]["eras"],
        genre_subgenres=raw["genre"]["subgenres"],
        intensity_adjectives=raw["intimate_gesture"]["intensity_adjectives"],
        delivery_nouns=raw["intimate_gesture"]["delivery_nouns"],
        motion_verbs=raw["dynamic_tension"]["motion_verbs"],
        musical_objects=raw["dynamic_tension"]["musical_objects"],
        environments=raw["sensory_bridge"]["environments"],
        sensory_mediums=raw["sensory_bridge"]["mediums"],
        emotions=raw["emotional_anchor"]["emotions"],
        emotional_arcs=raw["emotional_anchor"]["arcs"],
    )

@dataclass
class Persona:
    name: str
    base_style: str
    genre_affinity: List[str]
    gesture_affinity: List[str]
    emotional_traits: List[str]

def load_personas(path: Path) -> Dict[str, Persona]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    personas: Dict[str, Persona] = {}
    for name, values in raw.items():
        personas[name] = Persona(
            name=name,
            base_style=values["base_style"],
            genre_affinity=values["genre_affinity"],
            gesture_affinity=values["gesture_affinity"],
            emotional_traits=values["emotional_traits"],
        )
    return personas

def validate_style_config(cfg: StyleConfig) -> None:
    if not cfg.genre_eras:
        raise ValueError("No genre eras configured.")
    if "negative" not in cfg.emotions:
        raise ValueError("Expected 'negative' emotional category.")
    # Add whatever invariants matter to you

STYLE_KEYS = [
    "genre_anchor",
    "intimate_gesture",
    "dynamic_tension",
    "sensory_bridge",
    "emotional_anchor",
]

def hamming_distance_components(a: Dict[str, str], b: Dict[str, str]) -> int:
    """
    Hamming distance over the 5 main components.
    Counts how many components are different.
    """
    return sum(
        1 for key in STYLE_KEYS
        if a.get(key) != b.get(key)
    )

# ============================================================================
# COMPONENT A: GENRE ANCHOR
# ============================================================================

GENRE_ERAS = [
    "lo-fi", "hyperpop", "darkwave", "ritual", "forest", "witch-house",
    "cinematic", "Gqom-tempo", "drone-doom", "emo-trap", "boom-bap",
    "vaporwave", "synthwave", "minimal-techno", "shoegaze", "baroque-pop",
    "chamber-pop", "swing-jazz", "post-rock", "ethno-fusion", "dub",
    "reggae", "isicathamiya", "phonk", "noir", "spacewestern", "membran-phonk",
    "sacred-harp", "tuvan-drone", "classical-fugue", "memphis-phonk",
    "gagaku", "glitch-jazz", "steppe-reggae", "trip-hop", "mbube-soul",
    "echo-pop", "epic-doom", "dark-orchestral"
]

GENRE_SUBGENRES = {
    "lo-fi": ["boom-bap", "chill-hops", "jazz-sample", "ambient-lo-fi"],
    "hyperpop": ["meltdown", "glitch-pop", "chaotic-euphoria", "maximalist"],
    "darkwave": ["electro", "post-punk", "cold-wave", "synth-goth"],
    "ritual": ["ambient", "chant-driven", "ethnic-fusion", "ceremonial"],
    "forest": ["folk", "field-recording", "ambient-nature", "ethereal"],
    "witch-house": ["lullaby", "haunted-lullaby", "dark-ambient", "occult-pop"],
    "cinematic": ["pop-ballad", "orchestral", "epic-swell", "underscore"],
    "Gqom-tempo": ["ritual", "urban-dance", "township-energy", "polyrhythmic"],
    "drone-doom": ["folk", "funeral-march", "processional", "apocalyptic"],
    "emo-trap": ["ballad", "atmospheric", "vulnerable-rap", "heartbreak"],
}

# ============================================================================
# COMPONENT B: INTIMATE GESTURE (Micro performance description)
# ============================================================================

INTENSITY_ADJECTIVES = {
    "energy": ["whispered", "hushed", "breathy", "deadpan", "fragile", "raspy",
               "sermon-like", "chant-like", "taunting", "half-spoken", "velvet",
               "drunken", "diary-like"],
    "texture": ["creaking", "crackled", "glassy", "metallic", "woody", "reedy",
                "nasal", "gravelly", "silky", "gritty"],
    "emotional": ["vulnerable", "defiant", "ecstatic", "mournful", "playful",
                  "haunted", "hypnotic", "urgent"],
}

DELIVERY_NOUNS = {
    "spoken": ["confessions", "murmurs", "mantras", "ranting-verses", "sermon-speech",
               "diary-lines", "story-rap", "talk-singing", "call-and-response-shouts",
               "half-recited-prayers", "spoken-word-poetry"],
    "sung": ["falsetto-runs", "lullaby-vocals", "chant-hooks", "wailing-phrases",
             "soprano-flights", "modal-drones", "harmonic-swells", "ululating-cries"],
    "hybrid": ["spoken-song", "parlando-scat", "murmured-chant", "breathed-melody",
               "whispered-rap", "sacred-rap", "deadpan-crooning"],
}

# ============================================================================
# COMPONENT C: DYNAMIC TENSION (Trajectory, motion, time)
# ============================================================================

MOTION_VERBS = [
    "blooming", "decaying", "spiraling", "smoldering", "crackling",
    "pulsing", "stuttering", "tidal", "collapsing", "unraveling",
    "coiling", "flickering", "swelling", "crashing", "dissolving",
    "melting", "crystallizing", "breathing", "dripping", "cascading",
    "layering", "stripping", "rupturing", "floating", "sinking",
    "ascending", "descending", "vibrating", "churning", "whirling"
]

MUSICAL_OBJECTS = {
    "harmonic": ["harmonies", "arpeggios", "inversions", "modal-shifts", "chords"],
    "percussive": ["808s", "hi-hats", "kick-patterns", "snares", "tom-rolls",
                   "cymbals", "tam-tam-swells", "gongs"],
    "melodic": ["synths", "strings", "pads", "basslines", "leads", "countermelodies"],
    "textural": ["drones", "noise-beds", "reverb-trails", "haze", "shimmer", "static"],
    "choral": ["choirs", "harmonies", "vocal-layers", "ghost-chants", "ensemble-swell"],
}

# ============================================================================
# COMPONENT D: SENSORY BRIDGE / SPACE (Macro, visual-to-audio mapping)
# ============================================================================

ENVIRONMENTS = [
    "forest", "subway-tunnel", "neon-alley", "cathedral", "basement",
    "attic", "empty-gym", "midnight-highway", "foggy-harbor", "abandoned-factory",
    "rain-soaked-street", "theater-stage", "caves", "rooftop", "ballroom",
    "warehouse", "beach-at-dusk", "burnt-field", "hospital-wing", "chapel",
    "desert", "glacier", "shipwreck", "carnival", "ruins"
]

SENSORY_MEDIUMS = {
    "visual-lens": ["VHS", "CRT-screen", "phone-camera", "neon-lens", "sepia-film",
                    "CCTV", "Polaroid", "film-noir", "thermal-imaging", "kaleidoscope"],
    "audio-effect": ["reverb", "echo", "haze", "glow", "blur", "shimmer", "hum",
                     "static", "shadow", "crackle", "whisper", "roar"],
}

# ============================================================================
# COMPONENT E: EMOTIONAL ANCHOR (Meaning, emotional arc)
# ============================================================================

EMOTIONS = {
    "negative": ["heartbreak", "longing", "regret", "dread", "paranoia",
                 "despair", "anguish", "numbness", "void", "dissolution"],
    "positive": ["relief", "euphoria", "ecstasy", "awe", "wonder", "exaltation"],
    "complex": ["nostalgia", "forgiveness", "defiance", "acceptance", "bittersweet",
                "haunted", "redemption", "melancholy", "wistful"],
}

EMOTIONAL_ARCS = {
    "musical": ["crescendo", "comedown", "afterglow", "drop", "dissolve",
                "echo", "undertow", "flare", "blackout", "fade-to-silence",
                "surge", "collapse", "catharsis"],
    "temporal": ["surrender", "quiet-resolve", "numb-aftermath", "haunted-calm",
                 "bittersweet-dawn", "last-stand", "exodus", "dawn-arrival",
                 "midnight-reckoning", "eternal-return"],
}

# ============================================================================
# YAML CONFIG LOADING
# ============================================================================

def load_style_config(path: str) -> Dict[str, Any]:
    """
    Load style component configuration from a YAML file.

    Expected top-level keys (if following our earlier plan):
      - genre_anchor
      - intimate_gesture
      - dynamic_tension
      - sensory_bridge
      - emotional_anchor

    Each section can carry whatever nested structure you prefer; the
    generator functions will look up specific keys with safe fallbacks.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Style config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_personas_config(path: str) -> Dict[str, Any]:
    """
    Load personas configuration from a YAML file.

    Expected structure (minimal):

      persona_name:
        base_style: "..."
        genre_affinity: [...]
        gesture_affinity: [...]
        emotional_traits: [...]

    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Personas config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Personas YAML must map persona_name → persona dict")
    return data

# Helper functions for the above

def set_global_style_config(cfg: Dict[str, Any]) -> None:
    global STYLE_CONFIG
    STYLE_CONFIG = cfg


def set_global_personas_config(cfg: Dict[str, Any]) -> None:
    global PERSONAS_CONFIG
    PERSONAS_CONFIG = cfg


# ============================================================================
# GENERATOR FUNCTIONS
# ============================================================================

import re
from typing import Mapping, Any, Optional
import random

def generate_genre_anchor(
    style_components: Mapping[str, Any],
    genre_hint: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Generate [GENRE ANCHOR] component from data in style_components.yaml.

    If genre_hint is provided, try to bias choice of era/subgenre toward
    entries that textually match the hint (case-insensitive, token-based).
    """
    if rng is None:
        rng = random

    genre_cfg = style_components.get("genre", {})
    eras = genre_cfg.get("eras")
    sub_map = genre_cfg.get("subgenres", {})
    fallback = genre_cfg.get("fallback_subgenres")

    if not eras:
        raise ValueError("No 'genre.eras' configured in style_components.yaml")

    # If no hint given, behave as before
    if not genre_hint:
        era = rng.choice(eras)
        candidates = sub_map.get(era, fallback)
        if not candidates:
            raise ValueError(
                f"No subgenres defined for era {era!r} and no 'genre.fallback_subgenres' configured."
            )
        subgenre = rng.choice(candidates)
        return f"{era} {subgenre}"

    # ------------------------------
    # Hint-aware path
    # ------------------------------
    hint = genre_hint.lower()
    # crude tokenization: split on non-alphanumerics
    tokens = [t for t in re.split(r"[^a-z0-9+]+", hint) if len(t) >= 3]

    def score_era(era_name: str) -> int:
        """Score an era by how many hint tokens appear in era or its subgenres."""
        era_l = era_name.lower()
        score = sum(1 for t in tokens if t in era_l)

        # also check subgenres
        for sub in sub_map.get(era_name, []):
            sub_l = sub.lower()
            score += sum(1 for t in tokens if t in sub_l)
        return score

    # Compute scores for all eras
    scored = [(era_name, score_era(era_name)) for era_name in eras]
    max_score = max(s for _, s in scored)

    if max_score > 0:
        # at least some match the hint; keep only best-scoring eras
        best_eras = [era_name for era_name, s in scored if s == max_score]
    else:
        # nothing matches → fall back to all eras
        best_eras = list(eras)

    era = rng.choice(best_eras)

    candidates = sub_map.get(era, fallback)
    if not candidates:
        raise ValueError(
            f"No subgenres defined for era {era!r} and no 'genre.fallback_subgenres' configured."
        )
    subgenre = rng.choice(candidates)
    return f"{era} {subgenre}"

def generate_intimate_gesture(
    style_components: Mapping[str, Any],
    rng: Optional[random.Random] = None,
) -> str:
    """
    Generate [INTIMATE GESTURE] component from data in style_components.yaml.

    Expected structure:

    style_components["intimate_gesture"]["intensity_adjectives"] -> dict[str, list[str]]
    style_components["intimate_gesture"]["delivery_nouns"] -> dict[str, list[str]]
    """
    if rng is None:
        rng = random

    ig_cfg = style_components.get("intimate_gesture", {})
    intensity_map = ig_cfg.get("intensity_adjectives") or {}
    delivery_map = ig_cfg.get("delivery_nouns") or {}

    if not intensity_map:
        raise ValueError(
            "No 'intimate_gesture.intensity_adjectives' configured in style_components.yaml"
        )
    if not delivery_map:
        raise ValueError(
            "No 'intimate_gesture.delivery_nouns' configured in style_components.yaml"
        )

    # Pick one intensity category, then one adjective from that category
    intensity_key = rng.choice(list(intensity_map.keys()))
    adjective = rng.choice(intensity_map[intensity_key])

    # Pick one delivery category, then one noun from that category
    delivery_key = rng.choice(list(delivery_map.keys()))
    delivery = rng.choice(delivery_map[delivery_key])

    return f"{adjective} {delivery}"

def generate_dynamic_tension(style_cfg: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate [DYNAMIC TENSION] component, using STYLE_CONFIG if provided.

    Falls back to the original MOTION_VERBS / MUSICAL_OBJECTS constants
    if no YAML config is available.
    """
    cfg = style_cfg or STYLE_CONFIG

    # From YAML, we expect:
    # dynamic_tension:
    #   motion_verbs: [...]
    #   musical_objects:
    #     harmonic: [...]
    #     percussive: [...]
    #     melodic: [...]
    #     textural: [...]
    if cfg and "dynamic_tension" in cfg:
        dt_cfg = cfg["dynamic_tension"] or {}
        motion_verbs = dt_cfg.get("motion_verbs", MOTION_VERBS)
        musical_objects_cfg = dt_cfg.get("musical_objects", MUSICAL_OBJECTS)
    else:
        motion_verbs = MOTION_VERBS
        musical_objects_cfg = MUSICAL_OBJECTS

    verb = random.choice(motion_verbs)

    # Pick a musical object from one of the configured categories
    category_key = random.choice(list(musical_objects_cfg.keys()))
    obj = random.choice(musical_objects_cfg[category_key])

    # e.g. "blooming harmonies", "crackling 808s"
    return f"{verb} {obj}"

def generate_sensory_bridge(style_cfg: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate [SENSORY BRIDGE / SPACE] component, using STYLE_CONFIG if present.

    Pattern: "<environment>-<medium> <descriptor>"
      e.g. "forest-haze reverb", "neon-alley-light shimmer"
    """
    cfg = style_cfg or STYLE_CONFIG

    if cfg and "sensory_bridge" in cfg:
        sb_cfg = cfg["sensory_bridge"] or {}
        environments = sb_cfg.get("environments", ENVIRONMENTS)
        sensory_mediums_cfg = sb_cfg.get("sensory_mediums", SENSORY_MEDIUMS)
        descriptors = sb_cfg.get("descriptors", ["reverb", "echo", "haze", "glow", "blur", "shimmer", "hum"])
    else:
        environments = ENVIRONMENTS
        sensory_mediums_cfg = SENSORY_MEDIUMS
        descriptors = ["reverb", "echo", "haze", "glow", "blur", "shimmer", "hum"]

    environment = random.choice(environments)

    medium_key = random.choice(list(sensory_mediums_cfg.keys()))
    medium = random.choice(sensory_mediums_cfg[medium_key])

    descriptor = random.choice(descriptors)

    return f"{environment}-{medium} {descriptor}"

def generate_emotional_anchor(style_cfg: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate [EMOTIONAL ANCHOR] component.

    Pattern examples:
      "aching slow-burn",
      "haunted sparkle-fade",
      "defiant tension-release"
    """
    cfg = style_cfg or STYLE_CONFIG

    if cfg and "emotional_anchor" in cfg:
        ea_cfg = cfg["emotional_anchor"] or {}
        emotions_cfg = ea_cfg.get("emotions", EMOTIONS)
        arcs_cfg = ea_cfg.get("arcs", EMOTIONAL_ARCS)
    else:
        emotions_cfg = EMOTIONS
        arcs_cfg = EMOTIONAL_ARCS

    emotion_key = random.choice(list(emotions_cfg.keys()))
    emotion = random.choice(emotions_cfg[emotion_key])

    arc_key = random.choice(list(arcs_cfg.keys()))
    arc = random.choice(arcs_cfg[arc_key])

    return f"{emotion} {arc}"

def generate_full_definition(
    style_cfg: Optional[Dict[str, Any]] = None,
    genre_hint: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate all five components for a complete track definition.
    Uses STYLE_CONFIG as default if no cfg is passed.
    Optionally biases the genre anchor using genre_hint.
    """
    cfg = style_cfg or STYLE_CONFIG

    return {
        "genre_anchor": generate_genre_anchor(cfg, genre_hint=genre_hint),
        "intimate_gesture": generate_intimate_gesture(cfg),
        "dynamic_tension": generate_dynamic_tension(cfg),
        "sensory_bridge": generate_sensory_bridge(cfg),
        "emotional_anchor": generate_emotional_anchor(cfg),
    }

def generate_three_part_sequence(
    style_cfg: Optional[Dict[str, Any]] = None,
    personas_cfg: Optional[Dict[str, Any]] = None,
    persona_name: Optional[str] = None,
    genre_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a 3-part metaphor chain:
      - intro
      - middle (mid-track lift / peak)
      - outro (decay / afterimage)

    Each part is a normal 5-component definition; we then assemble them into
    one Style of Music string with labels.
    """
    cfg = style_cfg or STYLE_CONFIG
    persona_db = personas_cfg or PERSONAS_CONFIG or {}

    roles = [
        ("intro",  "Intro:",             genre_hint),
        ("middle", "→ mid-track lift:",  genre_hint),
        ("outro",  "→ outro:",           genre_hint),
    ]

    segments = []

    for role, label, role_hint in roles:
        components = generate_full_definition(cfg, genre_hint=role_hint)

        if persona_name:
            components = get_persona_modified_components(
                persona_name,
                components,
                personas_cfg=persona_db,
            )

        assembled = assemble_suno_style(components)
        segments.append(
            {
                "role": role,
                "label": label,
                "components": components,
                "assembled": assembled,
            }
        )

    # Final single-line sequence for Suno “Style of Music”
    sequence_str = " ".join(
        f"{seg['label']} {seg['assembled']}" for seg in segments
    )

    return {
        "segments": segments,          # full detail per part
        "sequence_assembled": sequence_str,  # final line to paste into Suno
    }


def assemble_suno_style(components: Dict[str, str]) -> str:
    """Assemble components into a single Suno-compatible style phrase."""
    return ", ".join([
        components["genre_anchor"],
        components["intimate_gesture"],
        components["dynamic_tension"],
        components["sensory_bridge"],
        components["emotional_anchor"],
    ])

# Extended Five-Component Framework
# Including: Persona Integration, LLM Prompts, and Batch Generation

# ============================================================================
# PERSONA-AWARE GENERATION
# ============================================================================

# Sample personas from the personas.md file (extracted)
PERSONAS_DB = {
    "Elena Vorn": {
        "base_style": "Hybrid anthemic metal meets cinematic EDM with female power vocals",
        "genre_affinity": ["metal", "EDM", "symphonic", "electronicore"],
        "gesture_affinity": ["power-vocals", "layered", "harmonic"],
        "emotional_traits": ["triumphant", "intense", "inspiring"],
    },
    "Matt Stoned": {
        "base_style": "Onomatopoetic synth loops, male ghost vocals, glitch harmony",
        "genre_affinity": ["ambient-synth", "vaporwave", "glitch-pop"],
        "gesture_affinity": ["ghost-vocals", "robotic", "fragmented"],
        "emotional_traits": ["drifting", "spectral", "otherworldly"],
    },
    "Fiona Stoned": {
        "base_style": "Onomatopoetic synth loops, female ghost vocals, glitch harmony",
        "genre_affinity": ["ambient-synth", "vaporwave", "glitch-pop"],
        "gesture_affinity": ["ghost-vocals", "robotic", "fragmented"],
        "emotional_traits": ["drifting", "spectral", "otherworldly"],
    },
    "Emma Kraft": {
        "base_style": "Mechanical synthpop pulse with analog textures, vocoder layers",
        "genre_affinity": ["synthpop", "electronic", "krautrock"],
        "gesture_affinity": ["robotic", "vocoded", "minimalist"],
        "emotional_traits": ["futuristic", "neutral", "synthetic"],
    },
    "Wilds sisters": {
        "base_style": "Minimal techno meets circus waltz, playful vocal glitches",
        "genre_affinity": ["minimal-techno", "circus-waltz"],
        "gesture_affinity": ["playful", "glitched", "mechanical"],
        "emotional_traits": ["surreal", "hypnotic", "uneasy"],
    },
    "Alice Payne": {
        "base_style": "Horror synth, gothic metal, musique concrete",
        "genre_affinity": ["horror-synth", "gothic-metal", "experimental"],
        "gesture_affinity": ["dramatic", "layered", "hallucinatory"],
        "emotional_traits": ["intense", "surreal", "dreamy"],
    },
}

def get_persona_modified_components(
    persona_name: str,
    base_components: Dict[str, str],
    personas_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Optionally modify components to better match a persona's style.

    Priority:
      1) personas_cfg (YAML-loaded)
      2) PERSONAS_CONFIG (global)
      3) PERSONAS_DB (hardcoded fallback)
    """
    # Resolve persona DB source
    persona_db = personas_cfg or PERSONAS_CONFIG or PERSONAS_DB

    persona = persona_db.get(persona_name)
    if not persona:
        return base_components

    modified = base_components.copy()

    genre_affinity = persona.get("genre_affinity") or []
    gesture_affinity = persona.get("gesture_affinity") or []
    emotional_traits = persona.get("emotional_traits") or []

    # Light-touch heuristics — additive, not destructive.

    # 1) Genre anchor: bias toward persona's genres.
    if genre_affinity:
        tag = random.choice(genre_affinity)
        # e.g. "... with gqom-tempo undercurrent"
        modified["genre_anchor"] = f"{modified['genre_anchor']} with {tag} undercurrent"

    # 2) Intimate gesture & dynamic tension: inject gesture flavor words.
    if gesture_affinity:
        gesture_tag = random.choice(gesture_affinity)
        # e.g. "whispered confessions" → "whispered confessions, ghost-vocals"
        modified["intimate_gesture"] = f"{modified['intimate_gesture']}, {gesture_tag}"
        # Optional: also nudge dynamic_tension slightly
        modified["dynamic_tension"] = f"{modified['dynamic_tension']} over {gesture_tag} patterns"

    # 3) Emotional anchor: tint by persona traits.
    if emotional_traits:
        emo_tag = random.choice(emotional_traits)
        # e.g. "aching slow-burn" → "aching, triumphant slow-burn"
        modified["emotional_anchor"] = f"{emo_tag} {modified['emotional_anchor']}"

    return modified

# ============================================================================
# LLM PROMPT TEMPLATES
# ============================================================================

def create_llm_prompt_for_generation(
    genre_hint: Optional[str] = None,
    persona_name: Optional[str] = None,
    count: int = 5,
    sequence: bool = False,
) -> str:
    """
    Create a prompt for an LLM (ChatGPT, Claude, etc.) to generate
    five-component style definitions ("metaphors") or 3-part chain of metaphors.
    """

    prompt = []
    prompt.append(
        "You are a music style designer for Suno and/or Producer.ai services. "
        "You generate compact 'metaphor chains' to be used as Suno/Producer 'Style of Music'/'Music Prompt' strings."
    )

    # Optional persona/genre context
    if persona_name and PERSONAS_CONFIG and persona_name in PERSONAS_CONFIG:
        persona = PERSONAS_CONFIG[persona_name]
        prompt.append(f"\n## Persona: {persona_name}")
        prompt.append(f"Base style: {persona['base_style']}")
        prompt.append(f"Genre affinity: {', '.join(persona['genre_affinity'])}")
        prompt.append(f"Emotional traits: {', '.join(persona['emotional_traits'])}")
        prompt.append("Honor this persona's aesthetic in all outputs.")

    if genre_hint:
        prompt.append("\n## Genre / mood hint")
        prompt.append(genre_hint)

    # Shared definition of the five components
    prompt.append(
        "\n## Five-component metaphor chain\n"
        "Each style definition is composed of 5 parts, in this order:\n"
        "  [A] Genre Anchor      – core genre + sub-flavor (e.g. 'cinematic pop-ballad')\n"
        "  [B] Intimate Gesture  – vocal/lead behavior (e.g. 'whispered bedside confessions')\n"
        "  [C] Dynamic Tension   – motion/time (e.g. 'slow-bloom piano harmonies')\n"
        "  [D] Sensory Bridge    – space / environment (e.g. 'bedroom-lamp reverb haze')\n"
        "  [E] Emotional Anchor  – emotional arc as metaphor (e.g. 'fragile-hope heartbeat')\n"
    )

    if not sequence:
        # ---- SINGLE DEFINITION MODE (CURRENT BEHAVIOR) ----
        prompt.append(
            "\n## Task\n"
            f"Generate {count} distinct single-line style definitions.\n"
            "Each definition should be a single comma-separated chain of the 5 parts:\n"
            "  [A], [B], [C], [D], [E]\n\n"
            "Example format:\n"
            "  Cinematic pop-ballad, whispered bedside confessions, "
            "slow-bloom piano harmonies, bedroom-lamp reverb haze, fragile-hope heartbeat\n\n"
            "Do not number the lines. Output exactly one line per style."
        )
    else:
        # ---- SEQUENCE MODE (INTRO / MID / OUTRO) ----
        prompt.append(
            "\n## Task (3-part sequences)\n"
            f"Generate {count} distinct 3-part style SEQUENCES.\n"
            "Each SEQUENCE describes one track with three time-slices:\n"
            "  • Intro\n"
            "  • Mid-track lift (peak / chorus / expansion)\n"
            "  • Outro (decay / afterimage)\n\n"
            "For each SEQUENCE:\n"
            "  - The Intro, Mid, and Outro are each their own five-component chain [A..E].\n"
            "  - Use the same overall aesthetic, but let the three parts evolve logically.\n"
            "  - Keep everything in one paragraph, tagged by section labels.\n\n"
            "Use this exact formatting style:\n"
            "  Intro:  [A_intro], [B_intro], [C_intro], [D_intro], [E_intro]; "
            "Mid: [A_mid], [B_mid], [C_mid], [D_mid], [E_mid]; "
            "Outro: [A_outro], [B_outro], [C_outro], [D_outro], [E_outro]\n\n"
            "Concrete example:\n"
            "  Intro: Cinematic pop-ballad, whispered bedside confessions, "
            "slow-bloom piano harmonies, bedroom-lamp reverb haze, fragile-hope heartbeat; "
            "Mid: Cinematic pop-ballad, voice opens into soaring octave leaps, "
            "strings unfurl in tidal arcs, hall-reverb wide shot, heartbreak crescendo; "
            "Outro: Cinematic pop-ballad, melody dissolves into single-note echoes, "
            "tape-warped piano afterimage, quiet streetlight loneliness, exhausted heart afterglow.\n\n"
            "Do not number the sequences. Output exactly one line per sequence."
        )

    return "\n".join(prompt)

def create_llm_prompt_for_refinement(
    initial_definition: str,
    refinement_goal: str
) -> str:
    """
    Create a prompt to refine an existing definition toward a specific goal.
    """
    
    return f"""You are refining a Suno track style definition using the FIVE-COMPONENT FRAMEWORK.

## CURRENT DEFINITION
"{initial_definition}"

## REFINEMENT GOAL
{refinement_goal}

Using the framework rules:
- A = Genre Anchor (era + subgenre)
- B = Intimate Gesture (adjective + delivery noun)
- C = Dynamic Tension (motion verb + musical object)
- D = Sensory Bridge (location + lens + effect)
- E = Emotional Anchor (emotion + arc)

Please:
1. Identify which components need adjustment
2. Propose refined versions of those components
3. Provide the new assembled definition

Generate two versions: no longer than 200 characters for Suno models v4.0 and earlier, and no more than 1000 characters for Suno models v4.5 and later.
"""

# ============================================================================
# BATCH GENERATION WITH EXPORT
# ============================================================================

def generate_batch_definitions(
    count: int = 10,
    persona_name: Optional[str] = None,
    export_format: str = "markdown",
    style_cfg: Optional[Dict[str, Any]] = None,
    personas_cfg: Optional[Dict[str, Any]] = None,
    genre_hint: Optional[str] = None,
    min_hamming: int = 0,
    max_attempts: int = 20,
) -> str:
    cfg = style_cfg or STYLE_CONFIG
    persona_db = personas_cfg or PERSONAS_CONFIG or {}

    definitions = []
    component_list: List[Dict[str, str]] = []  # keep raw components for distance

    for i in range(count):
        best_candidate = None
        best_min_distance = -1

        for attempt in range(max_attempts):
            components = generate_full_definition(cfg, genre_hint=genre_hint)

            if persona_name:
                components = get_persona_modified_components(
                    persona_name,
                    components,
                    personas_cfg=persona_db,
                )

            # If this is the first one, accept immediately
            if not component_list:
                best_candidate = components
                best_min_distance = len(STYLE_KEYS)
                break

            # Compute distance to existing ones
            distances = [
                hamming_distance_components(components, prev)
                for prev in component_list
            ]
            min_dist = min(distances)

            # If we meet the threshold, accept
            if min_dist >= min_hamming:
                best_candidate = components
                best_min_distance = min_dist
                break

            # Otherwise keep track of "best so far" in case we fail all attempts
            if min_dist > best_min_distance:
                best_candidate = components
                best_min_distance = min_dist

        # fallback: if we never met the threshold, take the best we saw
        if best_candidate is None:
            best_candidate = components  # extremely unlikely, but safe

        component_list.append(best_candidate)

        assembled = assemble_suno_style(best_candidate)
        definitions.append({
            "index": i + 1,
            "components": best_candidate,
            "assembled": assembled,
            "min_distance_in_batch": best_min_distance,
        })

    # Format output
    if export_format == "markdown":
        output = f"# Generated Suno Track Definitions\n\n"
        if persona_name:
            output += f"**Persona:** {persona_name}\n\n"
        output += f"**Generated:** {count} definitions\n\n---\n\n"
        
        for d in definitions:
            output += f"## Definition {d['index']}\n\n"
            output += f"```\n"
            output += f"[A] Genre Anchor:       {d['components']['genre_anchor']}\n"
            output += f"[B] Intimate Gesture:   {d['components']['intimate_gesture']}\n"
            output += f"[C] Dynamic Tension:    {d['components']['dynamic_tension']}\n"
            output += f"[D] Sensory Bridge:     {d['components']['sensory_bridge']}\n"
            output += f"[E] Emotional Anchor:   {d['components']['emotional_anchor']}\n"
            output += f"```\n\n"
            output += f"**Assembled:**\n```\n{d['assembled']}\n```\n\n"
        
        return output
    
    elif export_format == "json":
        return json.dumps([d["assembled"] for d in definitions], indent=2)
    
    elif export_format == "csv":
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Index", "Assembled Definition", "Genre Anchor", "Intimate Gesture", 
                        "Dynamic Tension", "Sensory Bridge", "Emotional Anchor"])
        
        for d in definitions:
            writer.writerow([
                d["index"],
                d["assembled"],
                d["components"]["genre_anchor"],
                d["components"]["intimate_gesture"],
                d["components"]["dynamic_tension"],
                d["components"]["sensory_bridge"],
                d["components"]["emotional_anchor"],
            ])
        
        return output.getvalue()
    
    return str(definitions)

# ============================================================================
# BATCH SEQUENCES GENERATION
# ============================================================================

def generate_batch_sequences(
    count: int = 10,
    persona_name: Optional[str] = None,
    export_format: str = "markdown",
    style_cfg: Optional[Dict[str, Any]] = None,
    personas_cfg: Optional[Dict[str, Any]] = None,
    genre_hint: Optional[str] = None,
) -> str:
    """
    Generate a batch of 3-part metaphor sequences and format for output.
    """
    cfg = style_cfg or STYLE_CONFIG
    persona_db = personas_cfg or PERSONAS_CONFIG or {}

    sequences = []

    for i in range(count):
        seq = generate_three_part_sequence(
            style_cfg=cfg,
            personas_cfg=persona_db,
            persona_name=persona_name,
            genre_hint=genre_hint,
        )
        sequences.append(
            {
                "index": i + 1,
                **seq,
            }
        )

    if export_format == "markdown":
        output = "# Generated 3-Part Metaphor Sequences\n\n"
        if persona_name:
            output += f"**Persona:** {persona_name}\n\n"
        if genre_hint:
            output += f"**Genre hint:** {genre_hint}\n\n"
        output += f"**Generated:** {count} sequences\n\n---\n\n"

        for seq in sequences:
            output += f"## Sequence {seq['index']}\n\n"

            # components by section
            for seg in seq["segments"]:
                output += f"### {seg['label']} ({seg['role']})\n\n"
                c = seg["components"]
                output += "```\n"
                output += f"[A] Genre Anchor:       {c['genre_anchor']}\n"
                output += f"[B] Intimate Gesture:   {c['intimate_gesture']}\n"
                output += f"[C] Dynamic Tension:    {c['dynamic_tension']}\n"
                output += f"[D] Sensory Bridge:     {c['sensory_bridge']}\n"
                output += f"[E] Emotional Anchor:   {c['emotional_anchor']}\n"
                output += "```\n\n"
                output += f"**Assembled {seg['role']}:**\n"
                output += f"```\n{seg['assembled']}\n```\n\n"

            # final single-line style for Suno
            output += "**Full Style Sequence:**\n"
            output += "```text\n"
            output += seq["sequence_assembled"] + "\n"
            output += "```\n\n---\n\n"

        return output

    elif export_format == "json":
        # Only export the assembled single-line sequences in JSON
        return json.dumps(
            [seq["sequence_assembled"] for seq in sequences],
            indent=2,
        )

    elif export_format == "csv":
        import csv
        from io import StringIO

        out = StringIO()
        writer = csv.writer(out)
        writer.writerow(
            ["Index", "Persona", "GenreHint", "SequenceAssembled"]
        )
        for seq in sequences:
            writer.writerow(
                [
                    seq["index"],
                    persona_name or "",
                    genre_hint or "",
                    seq["sequence_assembled"],
                ]
            )
        return out.getvalue()

    return str(sequences)

# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Suno style-only definitions or LLM prompts via the five-component framework."
    )

    # What to do
    parser.add_argument(
        "--task",
        choices=["definitions", "sequence", "llm-generate", "llm-refine"],
        default="definitions",
        help=(
            "What to run:\n"
            "  definitions  - generate single 5-component style-only definitions\n"
            "  sequence     - generate 3-part metaphor sequences (intro/mid/outro)\n"
            "  llm-generate - print an LLM prompt to generate five-component styles\n"
            "  llm-refine   - print an LLM prompt to refine an existing style line\n"
        ),
    )
    # Config files (needed for 'definitions', optional for llm-* modes)
    parser.add_argument(
        "--styles",
        type=str,
        required=False,
        help="Path to style_components.yaml (defines genre, gesture, tension, sensory, emotional pools).",
    )
    parser.add_argument(
        "--personas",
        type=str,
        required=False,
        help="Path to personas.yaml (optional; used for persona bias and persona-aware LLM prompts).",
    )

    # Persona selection (used in both 'definitions' and 'llm-generate')
    parser.add_argument(
        "--persona",
        type=str,
        required=False,
        help="Persona name to bias generation toward (must exist in personas YAML if provided).",
    )
    
    # RNG seed to produce stable results for the same parameters
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Optional random seed for reproducible output.",
    )

    # Shared count (definitions + llm-generate)
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="How many definitions to generate, or how many the LLM should be asked to create.",
    )

    # Output format (only meaningful for 'definitions')
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "json"],
        default="markdown",
        help="Output format for batch definitions (ignored in llm-* tasks).",
    )

    # Extra knobs for LLM generation mode
    parser.add_argument(
        "--genre-hint",
        type=str,
        required=False,
        help="Optional free-form hint for LLM generation (e.g. 'ritual forest-folk, ambient, ghost lullaby').",
    )

    # Inputs for LLM refinement mode
    parser.add_argument(
        "--initial-definition",
        type=str,
        required=False,
        help="Existing style-only definition to refine (used in --task llm-refine).",
    )
    parser.add_argument(
        "--refinement-goal",
        type=str,
        required=False,
        help=(
            "Free-form direction for refinement (used in --task llm-refine), e.g. "
            "'Make it darker, more introspective; shift toward indie-folk noir'."
        ),
    )
    parser.add_argument(
        "--llm-sequence",
        action="store_true",
        help=(
            "When used with --task llm-generate: ask the LLM to generate "
            "3-part style sequences (Intro/Mid/Outro) instead of single definitions."
        ),
    )
    # If variability requested, set the desired Hamming distance
    parser.add_argument(
        "--min-diff",
        type=int,
        default=0,
        help=(
            "Minimum Hamming distance between styles in a batch (0–5). "
            "0 = no constraint; 5 = all components must differ from every previous."
        ),
    )

    args = parser.parse_args()
    
    # ------------------------------------------------------------
    # Optional RNG seeding for reproducibility
    # ------------------------------------------------------------
    if args.seed is not None:
        random.seed(args.seed)

    # ------------------------------------------------------------
    # Load configs (only actually required for 'definitions')
    # ------------------------------------------------------------
    style_cfg: Optional[Dict[str, Any]] = None
    personas_cfg: Optional[Dict[str, Any]] = None

    if args.styles:
        style_cfg = load_style_config(args.styles)
        set_global_style_config(style_cfg)

    if args.personas:
        personas_cfg = load_personas_config(args.personas)
        set_global_personas_config(personas_cfg)

    # ------------------------------------------------------------
    # Dispatch by task
    # ------------------------------------------------------------
    if args.task == "definitions":
        if style_cfg is None:
            parser.error("--styles is required when --task=definitions")

        output = generate_batch_definitions(
            count=args.count,
            persona_name=args.persona,
            export_format=args.format,
            style_cfg=style_cfg,
            personas_cfg=personas_cfg,
            genre_hint=args.genre_hint,
            min_hamming=args.min_diff,
        )

    elif args.task == "sequence":
        if style_cfg is None:
            parser.error("--styles is required when --task=sequence")

        output = generate_batch_sequences(
            count=args.count,
            persona_name=args.persona,
            export_format=args.format,
            style_cfg=style_cfg,
            personas_cfg=personas_cfg,
            genre_hint=args.genre_hint,
        )

    elif args.task == "llm-generate":
        # No need for style/persona configs here: we just emit a text prompt
        output = create_llm_prompt_for_generation(
            genre_hint=args.genre_hint,
            persona_name=args.persona,
            count=args.count,
            sequence=args.llm_sequence,
        )

    elif args.task == "llm-refine":
        if not args.initial_definition:
            parser.error("--initial-definition is required when --task=llm-refine")
        if not args.refinement_goal:
            parser.error("--refinement-goal is required when --task=llm-refine")

        output = create_llm_prompt_for_refinement(
            initial_definition=args.initial_definition,
            refinement_goal=args.refinement_goal,
        )

    else:
        parser.error(f"Unknown task: {args.task}")

    print(output)

if __name__ == "__main__":
    main()
