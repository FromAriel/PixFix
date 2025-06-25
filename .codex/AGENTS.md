---

<!-- 
###############################################################################
# 🧠  Codex Agent Workspace – Tooling Contract & Guide (HIDDEN HEADER)
# Godot 4.4.1 · Headless · CI‑safe · .NET 8 SDK + Godot‑mono included
###############################################################################
-->

```text
###############################################################################
# 🧠  Codex Agent Workspace – Tooling Contract & Guide
# Godot 4.4.1 · Headless · CI‑safe · .NET 8 SDK + Godot‑mono included
###############################################################################
```

> \[!IMPORTANT]
>
> * **Indentation:** Always 4 spaces in `.gd`, `.gdshader`, `.cs`. Never use tabs.
> * `gdlint` expects `class_name` **before** `extends`.

---

## ──── SECTION: GODOT FIRST-TIME SETUP ────

1. **Use the built-in Godot CLI**: `/usr/local/bin/godot` (default in this image).
   To override, export `GODOT=/full/path/to/godot`.

2. **Import pass** – warm caches & create `global_script_class_cache.cfg`:

   ```bash
   godot --headless --editor --import --quit --path .
   ```

3. **Parse all GDScript**:

   ```bash
   godot --headless --check-only --quit --path .   # path MUST be repo root
   ```

4. **Build C#/Mono** (auto-skips if no `*.sln`):

   ```bash
   dotnet build > /tmp/dotnet_build.log
   tail -n 20 /tmp/dotnet_build.log
   ```

   * **Exit 0** ⇒ project is clean.
   * **Non‑zero** ⇒ inspect error lines and fix.

**Repeat steps 2–4 after any edit until all return 0.**

---

## ──── SECTION: PATCH HYGIENE & FORMAT ────

```bash
# Auto‑format changed .gd
.codex/fix_indent.sh $(git diff --name-only --cached -- '*.gd')

# Optional extra lint
gdlint $(git diff --name-only --cached -- '*.gd') || true

# C# style check
dotnet format --verify-no-changes || {
  echo 'C# code‑style violations detected.'; exit 1; }
```

* No tabs, no syntax errors, no style violations before commit.

---

## ──── SECTION: GODOT VALIDATION LOOP (CI) ────

```bash
godot --headless --editor --import --quit --path .   # refresh cache
godot --headless --check-only --quit --path .        # parse .gd
dotnet build > /tmp/dotnet_build.log                 # compile C# (auto-skip)
```

* For other languages: use the appropriate headless validation tools and skip validation if not applicable.

**Optional tests:**

```bash
godot --headless -s res://tests/          # GDScript tests
dotnet test                               # C#
cargo test | go test ./... | bun test     # others if present
```

---

## ──── SECTION: QUICK CHECKLIST ────

```
apply_patch
├─ gdformat  --use-spaces=4 <changed.gd>
├─ gdlint    <changed.gd> (non‑blocking)
├─ godot --headless --editor --import  --quit --path .
├─ godot --headless --check-only       --quit --path .
├─ dotnet build > /tmp/dotnet_build.log
└─ tail -n 20 /tmp/dotnet_build.log  →  ✔ commit / ✘ fix
```

---

## ──── SECTION: WHY THIS MATTERS ────

* `--import` is the **only** way to build Godot’s script-class cache.
* CI **skips** the import when no `main_scene` is set, so fresh repos won’t fail.
* `--check-only` finds GDScript errors; `dotnet build` ensures C# compiles.
  Together, these guarantee the project builds headless on any clean machine.

> **TL;DR:** Run the three headless commands. Exit 0 ⇒ good. Else, fix & rerun.

---

## ──── ADDENDUM: BUILD‑PLAN RULE SET ────

1. **Foundation first** – scaffolding (data models, interfaces, utils) is built before high-level features. CI fails fast if missing.
2. **Design principles** – data-driven, modular, extensible, compartmentalized. Follow each language’s canonical formatter (PEP 8, rustfmt, go fmt, gdformat, etc.).
3. **Indentation** – spaces-only except where a language **requires** tabs (e.g., `Makefile`). Keep tabs localized to that file type.
4. **Header-comment block** – for files that support comments, prepend:

   ```
   ###############################################################
   # <file path>
   # Key funcs/classes: • Foo – does X
   # Critical consts    • BAR – magic value
   ###############################################################
   ```

   Skip for formats with no comments (JSON, minified assets).
5. **Language-specific tests** – run `cargo test`, `go test`, `bun test`, etc., when present.

---

## ──── ADDENDUM: gdlint CLASS-ORDER WARNINGS ────

`gdlint` 4.x enforces **class‑definitions‑order**
(tool → `class_name` → `extends` → signals → enums → consts → exports → vars).

If it becomes noisy:

* Reorder clauses to match the list, or
* Suppress in file – `# gdlint:ignore = class-definitions-order`, or
* Customize via `.gdlintrc`, or
* Pin `gdtoolkit==4.0.1`.

CI runs `gdlint` **non-blocking**; treat warnings as advice until you’re ready to enforce them strictly.

---

```text
###############################################################################
# End of Codex Agent Workspace Guide
###############################################################################
```

---

**This Documents format ensures:**

* **ASCII headers** are preserved for style and grep-ability.
* **\[!IMPORTANT]** callouts highlight rules where required and supported.
* **Section separators** use bold, readable formatting.
* **All code and bash snippets** are easy to copy.
* **Machine parsing** is straightforward—each section is clearly delimited.

Let the USER know if you want any further tweaks.
