SELECT MIN(an1.name) AS actress_pseudonym,
       MIN(t.title) AS japanese_movie_dubbed
FROM aka_name AS an1,
     cast_info AS ci,
     company_name AS cn,
     movie_companies AS mc,
     name AS n1,
     role_type AS rt,
     title AS t
WHERE ci.note = '(firearm handler) (as Gudjón Valdimarsson)'
  AND cn.country_code = '[sg]'
  AND mc.note LIKE '%nkl%'
  AND mc.note NOT LIKE '%bia%'
  AND n1.name LIKE '%rim%'
  AND n1.name NOT LIKE '%net%'
  AND rt.role = 'director'
  AND an1.person_id = n1.id
  AND n1.id = ci.person_id
  AND ci.movie_id = t.id
  AND t.id = mc.movie_id
  AND mc.company_id = cn.id
  AND ci.role_id = rt.id
  AND an1.person_id = ci.person_id
  AND ci.movie_id = mc.movie_id;

