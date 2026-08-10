/* Проверка готовности панели в настоящем браузере, а не по чтению разметки.
 *
 * Четыре из пяти проверок готовности TASK-13 можно сделать только по отрисованному
 * документу: «сколько величин видно в покое» — это про DOM после отрисовки, а не про
 * шаблон, и «СТОП виден на каждом экране» тоже.
 *
 * Запуск (из каталога panel/):
 *   python3 -m http.server 8000 &
 *   node --experimental-default-type=module check.mjs
 *
 * Playwright ставится глобально; если импорт не находится, укажите путь:
 *   PLAYWRIGHT=$(npm root -g)/playwright/index.mjs node check.mjs
 */
/* Playwright может стоять глобально: тогда обычный импорт по имени его не находит.
   Путь берётся из PLAYWRIGHT, и это сказано в подсказке выше. */
const { chromium } = await import(process.env.PLAYWRIGHT || "playwright");

const URL = process.env.PANEL_URL || "http://localhost:8000/index.html";
const KEYS = ["1", "2", "3", "4", "5", "6", "7"];

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || "/opt/pw-browsers/chromium",
});
const page = await browser.newPage({ viewport: { width: 1600, height: 950 } });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));

await page.goto(URL);
await page.waitForSelector("#s-live .value");

const fails = [];
const say = (ok, text) => {
  console.log((ok ? "  " : "! ") + text);
  if (!ok) fails.push(text);
};

/* 1. В покое на первом экране не более шести величин. */
const values = await page.$$eval("#s-live .value", (e) => e.length);
say(values <= 6, `величин в покое на первом экране: ${values} (предел 6)`);

/* 2. Главное на экране занимает большую часть площади. */
const share = await page.evaluate(() => {
  const m = document.getElementById("mirrorwrap").getBoundingClientRect();
  const s = document.getElementById("s-live").getBoundingClientRect();
  return m.width / s.width;
});
say(share >= 0.6, `главный объект занимает ${(share * 100).toFixed(0)} % ширины экрана`);

/* 3. Переключение экранов с клавиатуры, Esc возвращает на первый. */
const switched = [];
for (const k of KEYS) {
  await page.keyboard.press(k);
  await page.waitForTimeout(90);
  switched.push(await page.$$eval(".screen.sel", (e) => e.length === 1 ? e[0].id : "?"));
}
say(new Set(switched).size === KEYS.length,
    `цифрами открывается ${new Set(switched).size} разных экранов из ${KEYS.length}`);
await page.keyboard.press("Escape");
await page.waitForTimeout(90);
say(await page.$eval(".screen.sel", (e) => e.id) === "s-live",
    "Esc возвращает на первый экран");

/* 4. СТОП виден на каждом экране. */
const stop = [];
for (const k of KEYS) {
  await page.keyboard.press(k);
  await page.waitForTimeout(60);
  stop.push(await page.isVisible("#stop"));
}
say(stop.every(Boolean), `СТОП виден на ${stop.filter(Boolean).length} экранах из ${KEYS.length}`);

/* 5. Накладки выключены по умолчанию и включаются по одной. */
await page.keyboard.press("1");
await page.waitForTimeout(90);
const before = await page.$eval("#overlaysvg", (e) => e.childElementCount);
await page.keyboard.press("g");
await page.waitForTimeout(90);
const after = await page.$eval("#overlaysvg", (e) => e.childElementCount);
say(before === 0 && after === 1,
    `накладок по умолчанию ${before}, после одной клавиши ${after}`);

/* 6. Конфигуратор открывается поверх любого экрана и предупреждает про форк. */
await page.keyboard.press("4");
await page.keyboard.press("p");
await page.waitForTimeout(150);
say(await page.isVisible("#cfg"), "конфигуратор открывается поверх экрана «Память»");
await page.hover("#cfg-b table");
await page.waitForTimeout(120);
const warn = (await page.textContent("#warn")) || "";
say(warn.includes("форк") || warn.includes("журнал форкается"),
    `касание блока Б предупреждает: «${warn.slice(0, 60)}…»`);

say(errors.length === 0, `ошибок в консоли: ${errors.length}` +
    (errors.length ? " — " + errors.join("; ") : ""));

await browser.close();
console.log(fails.length ? `\nне прошло: ${fails.length}` : "\nвсе проверки прошли");
process.exit(fails.length ? 1 : 0);
