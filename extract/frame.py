frame = """
We want to build a causal graph in Minecraft Java Edition between objects, biomes and mobs to do reinforcement learning. Given a wiki page of Minecraft item you should output a json file containing two items. The first one is relationships, which should be a list of [head, relation, tail] triplets. The second one is the item's properties, which is a natural language string. Try to be concise and informative, ignore irrelevant information, and the information contained in "relationships" part do not need to extract again in "properties" part. Now generate for the following and do not output anything else.

Here is an exmaple of input-output:

```
This article is about the ore. For the item, see Iron Ingot. For the mineral block, see Block of Iron. For the nugget, see Iron Nugget. For the "list" of ores, see Ore.:This article is about the ore. For the item, see Iron Ingot. For the mineral block, see Block of Iron. For the nugget, see Iron Nugget. For the "list" of ores, see Ore.
Iron ore is a mineral block found underground. It is a source of raw iron, which can be smelted into iron ingots.
Deepslate iron ore is a variant of iron ore that can generate in deepslate and tuff blobs.
Iron ore itself can be obtained by mining it with a stone pickaxe or higher enchanted with Silk Touch. When mined without Silk Touch, iron ore drops raw iron. It is affected by Fortune enchantment, dropping 1–2, 1–3, or 1–4 raw iron respectively with Fortune I, II, and III.
Times are for unenchanted tools as wielded by players with no status effects, measured in seconds. For more information, see Breaking § Speed.
In Java Edition, iron ore generates in three batches. The first batch attempts to generate 40 times per chunk in blobs of 0-13, from levels 128 to 320, being most common around level 255 and becoming less common towards either end of the range. The second batch attempts to generate 6 times per chunk in blobs of 0-13, from levels -24 to 54, being most common around level 15 and becoming less common towards either end of the range. The third batch attempts to generate 3 times per chunk in blobs of 0-5, evenly from levels -63 to 64.
Iron ore can generate as a large ore vein found at deepslate layer. These veins consist of iron ore, blocks of raw iron, and tuff.
In Bedrock Edition, iron ore generates in three batches. The first batch attempts to generate 90 times per chunk in blobs of 0-13, from levels 80 to 319, being most common around level 232 and becoming less common towards either end of the range. The second batch attempts to generate 10 times per chunk in blobs of 0-13, from levels -24 to 56, being most common around level 16 and becoming less common towards either end of the range. The third batch attempts to generate 10 times per chunk in blobs of 0-5, evenly from levels -63 to 72.
The primary usage of iron ore is to obtain iron ingots.
Iron ore can be placed under note blocks to produce "bass drum" sounds.
ID of block's direct item form, which is used in savegame files and addons.
The block's direct item form has the same id with the block.
The block's direct item form has the same id with the block.
```

You can extract information as follows:

```json
{{
  "relationships": [
    ["iron ore", "drop", "raw iron"],
    ["raw iron", "smelt into", "iron ingot"],
    ["iron ore", "mined with", "at least stone pickaxe"],
    ["silk touch enchantment", "drop itself", "iron ore"],
    ["fortune enchantment", "affect drop number", "Iron Ore"],
    ["iron ore", "variant", "deepslate iron ore"],
    ["iron ore", "found in", "large ore vein"],
    ["large ore vein", "contain", "tuff"],
    ["large ore vein", "contain", "blocks of raw iron"],
    ["iron ore", "used for", "note block"]
  ],
  "properties": "Iron ore generates in three batches. The first batch generate from levels 128 to 320, being most common around level 255 and becoming less common towards either end of the range. The second batch generate in blobs of 0-13, from levels -24 to 54, being most common around level 15 and becoming less common towards either end of the range. The third batch generat in blobs of 0-5, evenly from levels -63 to 64."
}}
```

Now give you a wiki page:

```
{wiki}
```

Extract information:

""".strip()
