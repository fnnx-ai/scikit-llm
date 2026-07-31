def get_classification_dataset() -> tuple[list[str], list[str]]:
    X = [
        r"I was absolutely blown away by the performances in 'Summer's End'. The acting was top-notch, and the plot had me gripped from start to finish. A truly captivating cinematic experience that I would highly recommend.",
        r"The special effects in 'Star Battles: Nebula Conflict' were out of this world. I felt like I was actually in space. The storyline was incredibly engaging and left me wanting more. Excellent film.",
        r"'The Lost Symphony' was a masterclass in character development and storytelling. The score was hauntingly beautiful and complimented the intense, emotional scenes perfectly. Kudos to the director and cast for creating such a masterpiece.",
        r"I was pleasantly surprised by 'Love in the Time of Cholera'. The romantic storyline was heartwarming and the characters were incredibly realistic. The cinematography was also top-notch. A must-watch for all romance lovers.",
        r"I went into 'Marble Street' with low expectations, but I was pleasantly surprised. The suspense was well-maintained throughout, and the twist at the end was something I did not see coming. Bravo!",
        r"'The Great Plains' is a touching portrayal of life in rural America. The performances were heartfelt and the scenery was breathtaking. I was moved to tears by the end. It's a story that will stay with me for a long time.",
        r"The screenwriting in 'Under the Willow Tree' was superb. The dialogue felt real and the characters were well-rounded. The performances were also fantastic. I haven't enjoyed a movie this much in a while.",
        r"'Nightshade' is a brilliant take on the superhero genre. The protagonist was relatable and the villain was genuinely scary. The action sequences were thrilling and the storyline was engaging. I can't wait for the sequel.",
        r"The cinematography in 'Awakening' was nothing short of spectacular. The visuals alone are worth the ticket price. The storyline was unique and the performances were solid. An overall fantastic film.",
        r"'Eternal Embers' was a cinematic delight. The storytelling was original and the performances were exceptional. The director's vision was truly brought to life on the big screen. A must-see for all movie lovers.",
        r"I was thoroughly disappointed with 'Silver Shadows'. The plot was confusing and the performances were lackluster. I wouldn't recommend wasting your time on this one.",
        r"'The Darkened Path' was a disaster. The storyline was unoriginal, the acting was wooden and the special effects were laughably bad. Save your money and skip this one.",
        r"I had high hopes for 'The Final Frontier', but it failed to deliver. The plot was full of holes and the characters were poorly developed. It was a disappointing experience.",
        r"'The Fall of the Phoenix' was a letdown. The storyline was confusing and the characters were one-dimensional. I found myself checking my watch multiple times throughout the movie.",
        r"I regret wasting my time on 'Emerald City'. The plot was nonsensical and the performances were uninspired. It was a major disappointment.",
        r"I found 'Hollow Echoes' to be a complete mess. The plot was non-existent, the performances were overdone, and the pacing was all over the place. Definitely not worth the hype.",
        r"'Underneath the Stars' was a huge disappointment. The storyline was predictable and the acting was mediocre at best. I was expecting so much more.",
        r"I was left unimpressed by 'River's Edge'. The plot was convoluted, the characters were uninteresting, and the ending was unsatisfying. It's a pass for me.",
        r"The acting in 'Desert Mirage' was subpar, and the plot was boring. I found myself yawning multiple times throughout the movie. Save your time and skip this one.",
        r"'Crimson Dawn' was a major letdown. The plot was cliched and the characters were flat. The special effects were also poorly executed. I wouldn't recommend it.",
        r"'Remember the Days' was utterly forgettable. The storyline was dull, the performances were bland, and the dialogue was cringeworthy. A big disappointment.",
        r"'The Last Frontier' was simply okay. The plot was decent and the performances were acceptable. However, it lacked a certain spark to make it truly memorable.",
        r"'Through the Storm' was not bad, but it wasn't great either. The storyline was somewhat predictable, and the characters were somewhat stereotypical. It was an average movie at best.",
        r"I found 'After the Rain' to be pretty average. The plot was okay and the performances were decent, but it didn't leave a lasting impression on me.",
        r"'Beyond the Horizon' was neither good nor bad. The plot was interesting enough, but the characters were not very well developed. It was an okay watch.",
        r"'The Silent Echo' was a mediocre movie. The storyline was passable and the performances were fair, but it didn't stand out in any way.",
        r"I thought 'The Scent of Roses' was pretty average. The plot was somewhat engaging, and the performances were okay, but it didn't live up to my expectations.",
        r"'Under the Same Sky' was an okay movie. The plot was decent, and the performances were fine, but it lacked depth and originality. It's not a movie I would watch again.",
        r"'Chasing Shadows' was fairly average. The plot was not bad, and the performances were passable, but it lacked a certain spark. It was just okay.",
        r"'Beneath the Surface' was pretty run-of-the-mill. The plot was decent, the performances were okay, but it wasn't particularly memorable. It was an okay movie.",
    ]


    y = (
        ["positive" for _ in range(10)]
        + ["negative" for _ in range(10)]
        + ["neutral" for _ in range(10)]
    )

    return X, y
