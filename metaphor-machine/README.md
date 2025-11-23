# Metaphor Machine script

Python script to generate

- metaphors: five-step descriptive styles (for "Style of Music" in Suno or "Sound Prompt" in Producer.ai)
- 3-chain sequences of metaphors for structured style-only definitions
- LLM prompts to generate or enhance an existing style

## Concept

### Demo

Before you proceed reading, try rendering one of the two styles (no Lyrics field; in Suno, set "Style influence" to a value circa 55%, and "Weirdness" circa 55%-60%).

** Cinematic Pop-Ballad ** (use Suno model v5.0 for best results)
```
Cinematic pop-ballad, whispered bedside confessions, slow-bloom piano harmonies, bedroom-lamp reverb haze, fragile-hope heartbeat  
→ mid-track lift: voice opens into soaring octave leaps, strings unfurl in tidal arcs, hall-reverb wide shot, heartbreak crescendo  
→ outro: melody dissolves into single-note echoes, tape-warped piano afterimage, quiet streetlight loneliness
```

and/or

** Boom-Bap Mutation ** (use Suno models v4.5, v4.5+ or v5.0)
```
Intro: boom-bap mutation, glassy ululating-cries, spiraling noise-beds, ruins-shadow hum, melancholy midnight-reckoning
→ mid-track lift: noir mutation, velvet wailing-phrases, spiraling reverb-trails, rooftop-whisper blur, dissolution crescendo
→ outro: classical-fugue fusion, metallic chant-hooks, swelling leads, glacier-crackle reverb, void quiet-resolve
```

The above have been created using the Metaphor Machine script (metaphor-machine.py). The first one has an extremely high ratio of producing rich vocal hallucinations ("songs in unknown language"), the second one can produce a variety of texture-rich, catchy melodies.

### The Idea

#### The Plot approach

Typical user-provided style of music  slots (comma separated) are just short, dry lists of genres, styles and moods. Experiments show that providing plot-like mini-sequences result in very texture-rich, unusual and variable results.

One can treat “Style of Music” as a **compressed score note** with 4–5 slots, or as a **plot with 5 acts** (steps):

1. **Core anchor** – genre + maybe era/production
2. **Voice / lead behavior** – what the voice *does*, not just its timbre
3. **Texture & motion** – how harmony/beat evolves over time
4. **Space & medium** – room, environment, or camera metaphor
5. **Emotional outcome** – how it *feels* at the end (release / unresolved / triumphant, etc.)

##### A. Template to reuse

Something like:

> **[genre + substyle]**, [vocal or lead gesture], [texture or motion], [space / camera metaphor], [emotional endpoint]

Examples in abstract:

- “lo-fi boom-bap, hushed diary-rap, vinyl-pop drum ghosts, stairwell reverb haze, bittersweet resolve”
- “hyperpop sugar-rush, pitch-bent cartoon hooks, sidechain tidal synths, glitch-confetti breakdown, euphoric overload”

---

##### B. Applying it to specific popular genres

Here are some concrete “styles only” examples in that pattern, across different genres. Each one follows your winning recipe.

###### 1) Classic / boom-bap hip-hop

> **Boom-bap hip-hop**, close-mic story-rap, **dusty swung snares**, stairwell-echo ad-libs, city-midnight nostalgia  

- Anchor: boom-bap hip-hop  
- Vocal gesture: “close-mic story-rap”  
- Motion: “dusty swung snares” (rhythmic feel)  
- Space: “stairwell-echo”  
- Emotion: “city-midnight nostalgia”

###### 2) Modern trap / emo-rap

> **Emo-trap ballad**, breathy auto-tune confessions, **sub-bass swell and stutter**, bedroom-reverb glow, blue-screen heartbreak  

You could swap pieces:

- Make it harsher → “…angry double-time triplets, cracked-phone-speaker distortion, parking-lot paranoia”
- Make it dreamy → “…soft 808 pillows, rain-on-window hi-hats, 3 AM drifting hope”

###### 3) EDM / festival melodic house

> **Melodic house**, wordless crowd-chant hooks, **rising sidechain waves**, festival-field stereo bloom, sunrise-drop catharsis  

You’ve still got:

- Genre anchor: melodic house
- Vocal gesture: “wordless crowd-chant hooks”
- Motion: “rising sidechain waves”
- Space: “festival-field … sunrise-drop”
- Emotional endpoint: “catharsis”

###### 4) Rock / post-rock

> **Post-rock surge**, almost-silent intro guitars, **slow tidal crescendo**, cathedral-hall delay trails, sky-opening finale  

You see the same pattern: genre → gesture → motion → space → endpoint.

###### 5) R&B / neo-soul

> **Neo-soul slow jam**, velvet falsetto runs, **lazy swing drum ghost-notes**, candle-lit studio haze, late-night forgiveness  

Or darker:

> **Alt-R&B noir**, smoke-worn alto murmurs, **minimal 808 heartbeats**, basement-lamp shimmer, unresolved desire  

###### 6) Metal / metalcore

> **Atmospheric metalcore**, scream-to-clean confession arcs, **double-kick storm under swelling chords**, abandoned-warehouse reverb, last-stand defiance  

Again: strong vocal gesture + motion + place.

---

##### C. How to pick metaphors so they actually *help* the model

The next idea: each comma should point at a **different musical dimension**:

1. **Genre / era / scene**  
   - “grunge ballad”, “UK drill”, “city-pop”, “Eurodance 2000s”, “bedroom indie folk”

2. **Vocal / lead behavior**  
   - Verbs + body language: “whispered confessions”, “half-spoken diary-tone”, “raspy shout-alongs”, “floaty falsetto leaps”, “deadpan talk-singing”

3. **Rhythm / motion / harmony**  
   - Words that imply **time**: “slow-bloom harmonies”, “stutter-step kicks”, “spiral chord progressions”, “tidal sidechain swells”, “falling arpeggio cascades”

4. **Space / camera / medium**  
   - Room + lens: “basement reverb”, “cathedral shimmer”, “phone-speaker crunch”, “neon-lens city blur”, “forest-reverb canopy”, “cassette-hiss vignette”

5. **Emotional endpoint or frame**  
   - “hopeful lift-off”, “numb aftermath”, “quiet resolve”, “bittersweet dawn,” “heartbreak crescendo”

---

##### D. A quick plug-and-play recipe one can keep using

When you want a new “styles only” line, try literally writing with placeholders:

> **[GENRE + SUBFLAVOR]**, [VOCAL GESTURE], [TEXTURE / MOTION], [SPACE / CAMERA METAPHOR], [EMOTIONAL ENDPOINT]

Example fills:

- “Guaracha club fire, shouted call-and-response hooks, ratcheting percussion ladders, sweat-haze dancefloor blur, delirious release”
- “Indie folk hush, near-silent porchlight vocals, fingerpicked heartbeat patterns, fireplace-room warmth, fragile comfort”
- “Gqom-tempo ritual, chant-like crowd stomps, stamping drum avalanches, warehouse-reverb thunder, possessed dance trance”

You don’t have to use all five slots every time (your three examples use four), but keeping **exactly one clear idea per comma** is what’s giving you those surprisingly emotional outputs.

#### Bringing this to life

##### 1. Turning this into a “metaphor machine”

Let’s explicitly treat these five as **independent knobs** you can dial per genre:

1. **Genre Anchor**
2. **Sensory Bridge**
3. **Dynamic Tension**
4. **Intimate–Expansive Juxtaposition**
5. **Emotional Anchor**

(We’ll keep Genre Anchor separate so the four conceptual layers sit on top of it.)

###### Base skeleton

A general pattern that encodes all four ideas:

> **[GENRE ANCHOR]**, [INTIMATE GESTURE], [DYNAMIC TENSION], [SENSORY BRIDGE / SPACE], [EMOTIONAL ANCHOR]

Where:

- **INTIMATE GESTURE** → how the vocal/lead behaves close-up  
- **DYNAMIC TENSION** → verbs + motion over time  
- **SENSORY BRIDGE / SPACE** → how the sound maps to visuals/space  
- **EMOTIONAL ANCHOR** → what emotional trajectory it implies

Now, to make this actually usable, let’s break each component into **recipe slots** and mini “word banks”.

---

##### 2. Dimension by dimension

###### A. Genre Anchor

This is your “gravity well”: it tells the model which record bin to search.

Pattern:

> **[era / vibe] [subgenre / hybrid]**

Examples for many of your playgrounds:

- “lo-fi boom-bap”, “emo-trap ballad”, “hyperpop meltdown”  
- “darkwave electro”, “ritual ambient”, “forest folk”, “witch-house lullaby”  
- “cinematic pop-ballad”, “Gqom-tempo ritual”, “drone-doom folk”

This part can stay fairly plain. The **magic** happens in the other four.

---

###### B. Intimate Gesture (micro, close-up)

**Goal:** Describe the *performance* like a person in a room, not a tag.

Pattern:

> [adjective] + [delivery noun]

Word bank seeds (you can expand):

- **Intensity/attitude**: whispered, hushed, breathy, deadpan, diary-like, drunken, fragile, raspy, sermon-like, chant-like, taunting, lullaby-soft, half-spoken  
- **Delivery nouns**: confessions, story-rap, murmurs, mantras, ranting verses, call-and-response shouts, falsetto runs, chant hooks, murmured prayers, diary lines

Examples:

- “whispered confessions”  
- “hushed diary-rap”  
- “velvet falsetto runs”  
- “chant-like spell vocals”  
- “deadpan surreal talk-singing”

This is where **Intimate** part of your “Intimate–Expansive” lives.

---

###### C. Dynamic Tension (time / motion)

**Goal:** Encode **trajectory**: up/down, bloom/decay, tension/release.

Patterns:

> [motion verb / metaphor] + [musical object]  
> [physical process] + [musical object]

Word bank seeds:

- **Verbs / processes**: blooming, decaying, spiraling, smoldering, crackling, pulsing, stuttering, tidal, collapsing, unraveling, coiling, flickering, swelling, crashing, dissolving  
- **Objects**: harmonies, synths, 808s, hi-hats, strings, pads, choirs, basslines, kick patterns, arpeggios, drones

Examples:

- “slow-bloom harmonies”  
- “brooding synth decay”  
- “stutter-step hi-hat ladders”  
- “tidal sidechain waves”  
- “collapsing string swells”  
- “coiled sub-bass pulses”

That’s your **Dynamic Tension** dimension.

---

###### D. Sensory Bridge / Space (macro, outer world)

**Goal:** Map sound → visual or spatial metaphor.

Pattern:

> [environment / medium] + [audio-ish noun]

Word bank seeds:

- **Environments**: forest, subway tunnel, neon alley, cathedral, basement, attic, empty gym, midnight highway, foggy harbor, abandoned factory, rain-soaked street  
- **Mediums / lenses**: VHS, CRT screen, phone-camera, neon-lens, sepia-film, CCTV, Polaroid, cassette  
- **Nouns**: reverb, echo, haze, glow, blur, shimmer, hum, static, shadow

Examples:

- “forest-reverb atmosphere”  
- “neon-lens dystopia”  
- “basement-reverb haze”  
- “cathedral-echo shimmer”  
- “phone-speaker crunch”  
- “VHS-flicker memory cloud”

This is where **Sensory Bridging** and the Expansive half live.

---

###### E. Emotional Anchor (what it *means*)

**Goal:** Tell the model what emotional *arc* this sound represents.

Two useful patterns:

> [emotion] + [musical metaphor]  
> [adjective] + [temporal/final-state noun]

Word bank seeds:

- **Emotions**: heartbreak, longing, regret, relief, euphoria, dread, paranoia, nostalgia, forgiveness, defiance, numbness, awe  
- **Musical metaphors**: crescendo, comedown, afterglow, drop, dissolve, echo, undertow, flare, blackout  
- **Final states**: surrender, catharsis, quiet resolve, numb aftermath, haunted calm, bittersweet dawn

Examples:

- “heartbreak crescendo”  
- “bittersweet dawn-comedown”  
- “numb aftermath glow”  
- “last-stand defiance”  
- “haunted calm landing”

This is exactly your **Emotional Anchoring Through Metaphor**.

---

##### 3. Assembling it: the “Metaphor Machine” recipe

###### Step-by-step generator

For a new style line:

1. **Pick Genre Anchor**  
   - e.g. “emo-trap ballad”

2. **Pick Intimate Gesture** (voice/lead)  
   - e.g. “breathy auto-tune confessions”

3. **Pick Dynamic Tension**  
   - e.g. “stutter-step hi-hat avalanches”

4. **Pick Sensory Bridge / Space**  
   - e.g. “bedroom-window rain reverb”

5. **Pick Emotional Anchor**  
   - e.g. “blue-screen heartbreak afterglow”

Combine:

> **Emo-trap ballad**, breathy auto-tune confessions, stutter-step hi-hat avalanches, bedroom-window rain reverb, blue-screen heartbreak afterglow

Same engine, different genre:

###### Example: Gqom-tempo ritual

1. Genre: “Gqom-tempo ritual”  
2. Intimate: “chant-like crowd stomps”  
3. Dynamic: “drum-quake build-and-crash patterns”  
4. Space: “warehouse-thunder reverb”  
5. Emotional: “possessed dance trance”

Result:

> **Gqom-tempo ritual**, chant-like crowd stomps, drum-quake build-and-crash patterns, warehouse-thunder reverb, possessed dance trance

###### Example: Ritual ambient × forest folk

1. Genre: “ritual forest-ambient”  
2. Intimate: “half-hummed folklore phrases”  
3. Dynamic: “slow-moss drone swelling”  
4. Space: “foggy pinewood echo canopy”  
5. Emotional: “ancient-comfort sleepwalking”

Result:

> **Ritual forest-ambient**, half-hummed folklore phrases, slow-moss drone swelling, foggy pinewood echo canopy, ancient-comfort sleepwalking

---

##### 4. Extending to “style sequences” (intro → middle → ending)

To encode evolution across the track, you can **chain 2–3 metaphors in one Style of Music** field, each one a compressed snapshot of a section:

For example, for a cinematic pop track:

> **Cinematic pop-ballad**, whispered bedside confessions, slow-bloom piano harmonies, bedroom-lamp reverb haze, fragile-hope heartbeat  
> → **mid-track lift**: voice opens into soaring octave leaps, strings unfurl in tidal arcs, hall-reverb wide shot, heartbreak crescendo  
> → **outro**: melody dissolves into single-note echoes, tape-warped piano afterimage, quiet streetlight loneliness

You’re still just writing text, but you’re giving the model **three time slices**:

- Intro state  
- Mid-track expansion  
- Outro decay

Same idea could be done more compactly with section keywords if you like:

> “Intro: hushed story-rap over dusty vinyl hush; Chorus: chant-like hooks explode in neon-echo hall; Outro: beat collapses into tape-hiss heartbeat and distant street sirens.”

The underlying machine is the same; you just apply it per “moment” in the track.

---

## Usage

The concept of creating plot-like sequences for a generative music service is implemented using two predefined configuration files

- style_components.yaml: genres choices gathered from Suno "Explore" page, the rest is created using common sense; adjust it if necessary
- personas.yaml: sample Suno personas definition, suitable for the purpose of Metaphor Machine

The script requires Python 3; "pyyaml" module may be required to install.

### How to invoke the script

The below try to use all the parameters; in real life, you might not need that.

** 0. Get self-explanatory help **

```bash
python metaphor-machine.py -h
```

** 1. Generate single metaphors **

```bash
python metaphor-machine.py \
  --task definitions \
  --styles style_components.yaml \
  --personas personas.yaml \
  --persona "Wilds sisters" \
  --count 6 \
  --genre-hint "phonk, memphis, old cassette" \
  --format markdown \
  --min-diff 3
```

** 2. Generate metaphor sequences **

```bash
python metaphor-machine.py \
  --task definitions \
  --styles style_components.yaml \
  --personas personas.yaml \
  --persona "Wilds sisters" \
  --count 6 \
  --genre-hint "phonk, memphis, old cassette" \
  --format markdown \
  --min-diff 3
```

** 3. Generate LLM prompt to generate metaphors **

```bash
python metaphor-machine.py \
  --task llm-generate \
  --persona "Alice Payne" \
  --genre-hint "industrial witch-house, dark pop" \
  --count 8 \
  --llm-sequence
```

** 4. Generate LLM prompt to refine styles **
```bash
python metaphor-machine.py \
  --task refine \
  --initial-definition "some style of music definition" \
  --refinement-goal "make it more vivid and brighter"
```

### Parameters explanation:

#### --task (definitions|sequence|llm-generate|llm-refine)

**Mandatory**. Chooses the operation mode.

One of "definitions" (create a list of metaphors), "sequence" (create a list of 3-chain metaphors), "llm-generate" (create a LLM prompt to request creating metaphors/metaphor chains), "llm-refine" (generate a LLM prompt to enhance an existing style)

#### --styles styles_components.yaml

**Mandatory** if `--task`  is `definitions` or `sequence`. Parameter is a path to YAML filename.

Read the enclosed working sample of style definitions (YAML file) and adjust it/generate your own as required. The enclosed one is a good starting point, matching Suno-recognized genres.

#### --personas personas.yaml

**Mandatory** if `--persona` parameter is used. Parameter is a path to YAML filename.

The enclosed sample refers to personas made publicly available on Suno. Define your own if/when necessary.

#### --count INT

**Optional**. Defaults to 5 if omitted. How many definitions or sequences to output. Ignored in llm-* tasks

#### --genre-hint STRING

**Optional** all tasks. Comma separated list of genres, for which the search/task should be performed.

If omitted, results in random selection of all the parameters.

#### --format (markdown|csv|json)

**Optional**. How to output the results (Markdown, CSV, JSON). Defaults to "markdown" if omitted.

#### --min-diff INT

**Optional**. Defaults to 0 if omitted. Sets the depth of Hamming distance between styles in a batch. 0 means no restriction; 5 means every term should differ from every previous one.

#### --llm-sequence

For `--llm-generate` task outputs the prompt to request sequences.

#### --initial-definition

**Mandatory** for `--llm-refine` task only. A string with free-form style of music definition.

#### --initial-definition

**Mandatory** for `--llm-refine` task only. A string with free-form explanation of how the style should be improved.

## Facts

The supplied style_components.json allows generating **156 × 806 × 900 × 3,850 × 575 ≈ 2.505 × 10¹⁴** different metaphors.

The supplied style_components.json allows generating **≈ 1.6 × 10⁴³** distinct intro–mid–outro sequences.

Have fun.
