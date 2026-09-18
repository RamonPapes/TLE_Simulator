import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import httpx

from passpredict import CelestrakTLESource, Location, SGP4Propagator, Observer
from passpredict.exceptions import CelestrakError
from passpredict.sources import parse_tle


class CelestrakOrgTLESource(CelestrakTLESource):
    """
    passpredict hardcodes the old celestrak.com domain, whose TLS certificate
    expired on 2026-04-12. Query celestrak.org instead, in 2-line TLE format
    (FORMAT=2LE), which omits the satellite name line.
    """
    BASE_URL = 'https://celestrak.org/NORAD/elements/gp.php'

    def _get(self, params: dict) -> list:
        r = httpx.get(self.BASE_URL, params=params, follow_redirects=True)
        r.raise_for_status()
        lines = [line for line in r.text.splitlines() if line.strip()]
        if not lines or lines[0].lower().strip() in ("no tle found", "no gp data found"):
            raise CelestrakError(f'Celestrak TLE not found for {params}')
        return lines

    def _query_tle_from_celestrak(self, satid: int = None):
        lines = self._get({'CATNR': satid, 'FORMAT': '2LE'})
        return parse_tle(lines[:2])

    def _query_tle_category_from_celestrak(self, category: str):
        lines = self._get({'GROUP': category, 'FORMAT': '2LE'})
        return [parse_tle(lines[i:i + 2]) for i in range(0, len(lines), 2)]


location = Location('Austin, TX', 30.2711, -97.7437, 0)
date_start = datetime.datetime.now(tz=ZoneInfo('America/Chicago'))
date_end = date_start + datetime.timedelta(days=10)
source = CelestrakOrgTLESource()
tle = source.get_tle(25544)  # International space station, Norad ID 25544
satellite = SGP4Propagator.from_tle(tle)
observer = Observer(location, satellite)
overpasses = observer.pass_list(date_start, limit_date=date_end)

print(f"TLE satid={tle.satid}  epoch={tle.epoch:%Y-%m-%d %H:%M:%S}")
print(tle.tle1)
print(tle.tle2)
print(f"\n{len(overpasses)} passagens sobre {location.name}\n")
for op in overpasses:
    print(f"{op.aos.dt:%Y-%m-%d %H:%M:%S %Z}  elev. max {op.tca.elevation:5.1f}°  "
          f"duracao {op.los.dt - op.aos.dt}")
