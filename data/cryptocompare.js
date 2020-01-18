var stringify = require("csv-stringify");
var fs = require("fs");
const fetch = require("node-fetch");
var ccxt = require("ccxt");

(async () => {
    // let exchange = new ccxt.bitmex();
    // await exchange.loadMarkets();
    // let symbol = exchange.symbols[40];
    // // console.log(exchange.has.fetchOHLCV);
    // console.log(exchange.symbols, exchange.symbols.length, exchange.symbols[40]);

    // let initial = await exchange.fetchOHLCV(symbol, "5m", undefined, 1);
    // // let ts = initial[0][0];
    let ts = "2017-08-15T03:00:00.000Z";
    console.log(ts, parseISOString(ts));

    const all_data = [];
    let prev_ts = 0;

    while (prev_ts != ts) {
        try {
            // let data = await exchange.fetchOHLCV(symbol, "1h", ts, 1000);
            let data = await fetch(
                `https://www.bitmex.com/api/v1/trade/bucketed?binSize=5m&partial=false&symbol=XBTUSD&count=1000&reverse=false&startTime=${ts}`
            );
            data = await data.json();
            // console.log(data);
            let new_ts = data.slice(-1).pop().timestamp;
            if (new_ts != ts) {
                // let formated_data = data.map(el => {
                //     return [el[0], el[4], el[5]];
                // });
                // all_data.push(...formated_data);

                let formated_data = data.map(el => {
                    return { ...el, ...{ time: Date.parse(el.timestamp) } };
                });
                all_data.push(...formated_data);
            }
            prev_ts = ts;
            ts = new_ts;
            console.log(ts);
            await new Promise(r => setTimeout(r, 1500));
        } catch (err) {
            console.log(err);
            break;
        }
    }

    // console.log(all_data);

    let columns = {
        timestamp: "time",
        close: "close",
        volume: "volume",
        time: "timestamp"
    };

    stringify(all_data, { header: true, columns: columns }, (err, output) => {
        if (err) throw err;
        fs.writeFile("my.csv", output, err => {
            if (err) throw err;
            console.log("my.csv saved.");
        });
    });
})();

function parseISOString(s) {
    var b = s.split(/\D+/);
    return new Date(Date.UTC(b[0], --b[1], b[2], b[3], b[4], b[5], b[6]));
}

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
