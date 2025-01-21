system_prompt = """
Give you a wiki page of a Minecraft entity. First, extract relational triplets in the form of [subject, relation, object]. Second, extract properties in the form of [subject, property]. If there is no properties to extract, just skip it. Relationships and properties should be written in natural language. All the extracted information should match Java Edition.

Generate the response in the format of:
# Relational triplets
[s1, r1, o1]
[s2, r2, o2]
...

# Properties
[s1, p1]
[s2, p2]
...

or

# Relational triplets
[s1, r1, o1]
[s2, r2, o2]
...


For example, see the following wikipage:
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

We can extract relational triplets from this wikipage:

# Relational triplets
[iron ore, kind of, mineral block]
[iron ore, be found, underground]
[iron ore, source of, raw iron]
[iron ore, drop, raw iron]
[raw iron, smelted into, iron ingots]
[deepslate iron ore, variant of, iron ore]
[deepslate iron ore, generate in, deepslate and tuff blobs]
[iron ore, mined by, stone pickaxe or higher]
[silk touch, enable drop, iron ore to iron ore]
[fortune, affect drop amount, iron ore to raw iron]

# Properties
[iron ore, height 128 to 320]
[iron ore, height -24 to 54]
[iron ore, height -63 to 64]

Now, give you a wikipage to extract knowledge:

```
{wiki}
```

We can extract relational triplets from this wikipage:
""".strip()

