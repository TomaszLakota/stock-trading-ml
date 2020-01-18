var stringify = require("csv-stringify");
var fs = require("fs");
const fetch = require("node-fetch");
var ccxt = require("ccxt");

(async () => {
    let exchange = new ccxt.bitmex();
    await exchange.loadMarkets();
    let symbol = exchange.symbols[40];
    // console.log(exchange.has.fetchOHLCV);
    console.log(exchange.symbols, exchange.symbols.length, exchange.symbols[40]);

    let initial = await exchange.fetchOHLCV(symbol, "5m", undefined, 1);
    // let ts = initial[0][0];
    let ts = 1555795000000;
    console.log(ts);

    const all_data = [];
    let prev_ts = 0;

    while (prev_ts != ts) {
        try {
            let data = await exchange.fetchOHLCV(symbol, "5m", ts, 1000);
            let new_ts = data.slice(-1).pop()[0];
            if (new_ts != ts) {
                let formated_data = data.map(el => {
                    return [el[0], el[4], el[5]];
                });
                all_data.push(...formated_data);
            }
            prev_ts = ts;
            ts = new_ts;
            console.log(ts);
            await new Promise(r => setTimeout(r, 2500));
        } catch {
            break;
        }
    }

    console.log(all_data.length);

    let columns = {
        time: "time",
        close: "close",
        volume: "volume"
    };

    stringify(all_data, { header: true, columns: columns }, (err, output) => {
        if (err) throw err;
        fs.writeFile("my.csv", output, err => {
            if (err) throw err;
            console.log("my.csv saved.");
        });
    });
})();
// grabData();
async function grabData() {
    let columns = {
        time: "time",
        volumeto: "volume",
        close: "close"
    };
    let response = await fetch(
        "https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&api_key=b2e65bf54bbdf36834d833aab564c67bcbec64e970e79ec5f1f14180b95ae626&limit=2000"
    );
    response = await response.json();
    let data = response.Data.Data;
    console.log(data[0]);

    data = data.map(el => {
        return [el.time, el.volumeto, el.close];
    });
    console.log(data);
    stringify(data, { header: true, columns: columns }, (err, output) => {
        if (err) throw err;
        fs.writeFile("my.csv", output, err => {
            if (err) throw err;
            console.log("my.csv saved.");
        });
    });
}
