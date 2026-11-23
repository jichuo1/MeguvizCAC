/**
 * In-browser Bilibili Wave-2 Era-Targeted & Zero-Overlap Multi-Tab Collector for MeguvizCAC v0.4.
 * Fixes Wave-1 multi-tab duplication & filename collisions via:
 *   1. Pre-seeded skip list (656 already-ingested BVIDs) + live localStorage union-merge + BroadcastChannel sync
 *   2. Unique per-tab chunk naming: bili_v04_<wave>_<runTag>_chunk_001.jsonl (never collides with Chrome (1)/(2))
 *   3. Three mathematically disjoint era presets ('pre_llm', 'early_llm', 'current_2026_news')
 * Strictly strips all author nicknames, UIDs, and IP locations; halts all workers on any -352/-412/-101/-403 gate.
 */
(function (root) {
  'use strict';

  const STORAGE_KEY = '__meguviz_v04_seen_bvids';
  const CHANNEL_NAME = '__meguviz_v04_bvid_sync';

  const INITIAL_SKIP_BVIDS = [
    'BV117ZYYXEce', 'BV11CY9zbE5v', 'BV11CZwYeEmP', 'BV11PZsY8EtB', 'BV11SZBYKEKe', 'BV11WZxYvE35',
    'BV11eZ8Y5EVi', 'BV11eZ8YLEZ6', 'BV11jZHY9EyS', 'BV11s4y1c7QV', 'BV127ZvYhEHb', 'BV12B4xziEgw',
    'BV12CZBYgEBi', 'BV12DZ6YVEkB', 'BV12VZHYdEQ4', 'BV12sZqYeEvD', 'BV12tZEYSEeU', 'BV12uZbYkEo5',
    'BV12wsm6CEic', 'BV13Apt6sEhK', 'BV13D421j7YJ', 'BV13JZ8YREua', 'BV13Ys16yEra', 'BV13bZpYvEBD',
    'BV13bZpYvEDr', 'BV13bZpYvEiZ', 'BV13fZEYbEd3', 'BV13kZhYZEQb', 'BV13uQrYzEMW', 'BV13yZhY4Ey6',
    'BV141Z7YEESX', 'BV141oXYGE6J', 'BV142Z7YmE4y', 'BV14DZJYXEEt', 'BV14HZqYiETU', 'BV14KZbYyEE6',
    'BV14eZCYQEgo', 'BV14tZzYgESy', 'BV152ZqYJEz2', 'BV153ZEYaEDW', 'BV153o9YsEgb', 'BV154Z4YpEeT',
    'BV154Z4YpEid', 'BV155ZEY8EcK', 'BV158oRYcE7F', 'BV159ZtY9EKf', 'BV159ZzYpEUk', 'BV15BZhYhE1j',
    'BV15DZzYsEP1', 'BV15FZGYyEdc', 'BV15PyszXE5S', 'BV15RZzYQETY', 'BV15VZEYFER6', 'BV15XdPYJE6P',
    'BV15ZZCYKE6f', 'BV15bZfY3E8B', 'BV15dovYrET1', 'BV15gZzYXE4N', 'BV15so4BgEgQ', 'BV15tZPYWEUh',
    'BV166ZtYaERo', 'BV167Z4YbEd4', 'BV169ZiYREdV', 'BV16AZrYNEwZ', 'BV16FZzYbECB', 'BV16WZJY3EwM',
    'BV16aZWYHEVv', 'BV16dZfYDEwC', 'BV16dZiYWEms', 'BV16gZpYmEqW', 'BV16io9YTEqH', 'BV16kZtYcEuP',
    'BV16oZBYhEZ3', 'BV16pZhYrEkc', 'BV16w411u7sH', 'BV173ZFYUEUc', 'BV174oRYAE77', 'BV17RZaYzEJx',
    'BV17TZEYPEug', 'BV17VZpY1ENA', 'BV17W4y1J7R4', 'BV17WZ4YsEiX', 'BV17aZhYvEiK', 'BV17bZhYqEY3',
    'BV17gZEYwE7J', 'BV17hZhYKEuJ', 'BV17q421w73e', 'BV17tZhYMEDq', 'BV17v4y1S7fE', 'BV18CEY6oEyH',
    'BV18EBLY8E3N', 'BV18FZbYUE3r', 'BV18FZbYUENq', 'BV18MZYYfEX3', 'BV18PAreAEAC', 'BV18XfTYoEbC',
    'BV18i4y1i7sT', 'BV18kSgBMEtJ', 'BV18kZ7YnEcm', 'BV18pZrYHEyt', 'BV18tZCYdEpc', 'BV199ZtYXEe9',
    'BV19AZWY3EHr', 'BV19JZCYwESj', 'BV19KZWYuE6W', 'BV19LfPYsEhG', 'BV19ZzVBtEHd', 'BV19jZ8YdExb',
    'BV19jfPY9EPg', 'BV19oZ8YbEPy', 'BV19uZtYXEC8', 'BV19vZ6YWE8G', 'BV1A2ofBjEEY', 'BV1AAfKYGE7v',
    'BV1ACZnY7Eb6', 'BV1AGZKYGE4n', 'BV1AKHDetEjV', 'BV1AQZ4YKEXu', 'BV1AWjBzuEdT', 'BV1AXbFezESG',
    'BV1AiZEY5Etd', 'BV1Ao4y1S7pu', 'BV1ArZnYUE9a', 'BV1B3ZnYuEDP', 'BV1B8ZJYTEPg', 'BV1BLZpY5E9n',
    'BV1BPZzYdEHp', 'BV1BUZaYkEkF', 'BV1BWet6FEWk', 'BV1BaZHYCEdM', 'BV1Bb4y1p7iB', 'BV1Bq4y1v7oN',
    'BV1BtZJYBE4G', 'BV1C8o9Y3Evf', 'BV1CAZpYpEzB', 'BV1CaZxYFEFG', 'BV1CervB3E8b', 'BV1CpZHYSExj',
    'BV1CtXqBVEh6', 'BV1Cx4y1s73T', 'BV1D9ZWYTEvm', 'BV1DAZHYYEoj', 'BV1DDZHYhE59', 'BV1DFpq67EB5',
    'BV1DJZ8Y9EtB', 'BV1DtZCYdEH8', 'BV1DzZ6YDEVJ', 'BV1E7oXYhEPv', 'BV1ED421j7eZ', 'BV1EKZnYkEqB',
    'BV1EPZEYKEvW', 'BV1EbZhYqEjd', 'BV1EeZcYQEDc', 'BV1EhZhYKEuU', 'BV1EiQ8YHEud', 'BV1EjoXYwECL',
    'BV1EnoXYbEu8', 'BV1ExZhYzECp', 'BV1F2ZnYLENk', 'BV1FFfwYZEM1', 'BV1FNAheQEqV', 'BV1FSZnYVE7P',
    'BV1FUoxYxEm2', 'BV1FV4y1C7Q2', 'BV1FWZkY6E6N', 'BV1FYAczhEt3', 'BV1FcZtYFESy', 'BV1FwoaBhEPN',
    'BV1G1ofYiENJ', 'BV1GDZqY2EAd', 'BV1GEZEYdEwn', 'BV1GWZnYcEYX', 'BV1GWZnYcEsu', 'BV1GXZAY7EWe',
    'BV1GnZnYAEU6', 'BV1GnZnYAEtn', 'BV1GnZnYAEwx', 'BV1GrofYuEEh', 'BV1GtZpYKEPo', 'BV1GyZYYNErW',
    'BV1H4SLY5EVJ', 'BV1H5Z4YhEsn', 'BV1H7ZbYTEwy', 'BV1HLZbYBEXg', 'BV1HTZpYSExJ', 'BV1HY411L78u',
    'BV1HpZhYrE2R', 'BV1HpZhYrELQ', 'BV1J6ZzYwEJU', 'BV1JNZnYVEQc', 'BV1JW4y1q7QV', 'BV1JWoRYKEHD',
    'BV1JcZtYFEDq', 'BV1JcqkBVEfR', 'BV1JhZcY9EFN', 'BV1JjZ4YFE26', 'BV1JjZsYHEHE', 'BV1JtZgYmEHD',
    'BV1JtzxBqEuj', 'BV1JwZ8YTECV', 'BV1KRZhYSE98', 'BV1KXZJYpEut', 'BV1KXZbYVEWb', 'BV1KcHP6yEx6',
    'BV1KcZwYPECW', 'BV1KnZGYcEyh', 'BV1Kr9gBCE67', 'BV1KzZCYJEFV', 'BV1L6ZvY1EEU', 'BV1L6ZvY1EMm',
    'BV1L7ZnYNEVV', 'BV1L7ZnYNEuH', 'BV1L8Z6YZEDS', 'BV1LHZpYgEPn', 'BV1LmKJ6eEL8', 'BV1Lo4y1U7MP',
    'BV1LvZhYYEso', 'BV1LwfhBbEWu', 'BV1LxZfYVEdT', 'BV1M2ZbY7EAc', 'BV1M4XhY3Ed6', 'BV1M6Z4YCEE7',
    'BV1MCfcYzEZG', 'BV1MKZbY1EBP', 'BV1MXZJYpEuN', 'BV1MbZnYoEk1', 'BV1MbfPY2EeG', 'BV1MgZhYDEt7',
    'BV1MjpZeZEHv', 'BV1MtZEYSEHD', 'BV1N2oXYrEVh', 'BV1N64y1Y768', 'BV1N7oiY4Eqn', 'BV1NBZnYDEi9',
    'BV1NLZHYxEdH', 'BV1NXfcYYEVq', 'BV1NZofYtEkp', 'BV1NcZeYWEZe', 'BV1NyoDYNEBq', 'BV1P3ZPYNETK',
    'BV1P8iwBMEbv', 'BV1P9RnBuEac', 'BV1PBZaYnErd', 'BV1PLZ6YMEe7', 'BV1PUZsY8E8m', 'BV1Pd4y1Q7bK',
    'BV1PhZtYzEnA', 'BV1Pho9YrEre', 'BV1PrZaYEEt1', 'BV1Pv4kzCEEM', 'BV1Pz4y1z7LS', 'BV1Q7ZtY7EBJ',
    'BV1QE9sBREpH', 'BV1QJoZYeExs', 'BV1QJodYdEWV', 'BV1QTZWYBE8W', 'BV1QV4y1G756', 'BV1QWZFYzEBd',
    'BV1QXZxY1EuR', 'BV1Qdc3zqEYz', 'BV1QhZwYkExK', 'BV1QvZBY1EYw', 'BV1QyZJYoEjB', 'BV1REFezMEa6',
    'BV1RF411s7EM', 'BV1RQZEYAEDd', 'BV1RXZEYaEpJ', 'BV1RXZtYWE9n', 'BV1RdZEY8E8T', 'BV1Rg411x7gk',
    'BV1RjYs65Ekq', 'BV1RoZ8YbE4g', 'BV1RoZnYqE5K', 'BV1Rt4y1L74a', 'BV1RxZPY9EFU', 'BV1S1oeYbEe1',
    'BV1SYXVBXEea', 'BV1SiZwYuEt6', 'BV1SrZ6YJEe1', 'BV1SsfwY6EKv', 'BV1SsfwY6EM7', 'BV1SuZHYbEkW',
    'BV1T14y1G7wm', 'BV1T2ZBYuEwV', 'BV1T2hR6EEie', 'BV1TLnB6uEkN', 'BV1TQZhYQESu', 'BV1TW4y1G72U',
    'BV1TeZnYWErJ', 'BV1TeZnYWEvP', 'BV1TfZcYvEs6', 'BV1ThZnYxEZi', 'BV1TnoQY6EX7', 'BV1TqZwYzED5',
    'BV1U7ZxYzEkC', 'BV1UE421w7YW', 'BV1UEo9YUE95', 'BV1UGZHY4EaZ', 'BV1USZJYWEGN', 'BV1UbZ8YmEP8',
    'BV1UeRWBUEH3', 'BV1Uks46REp2', 'BV1Ut4y1Z7fx', 'BV1V1ZcY5ESQ', 'BV1VAZHYeEud', 'BV1VLZ8YtE5W',
    'BV1VU4EeTERB', 'BV1VaoRYfEZH', 'BV1VfZEYbEri', 'BV1VmZqYUE87', 'BV1VqZhYwE1n', 'BV1W5fAYTEXS',
    'BV1W9ZbYkEwf', 'BV1WAZsYcERD', 'BV1WEfNYGEbz', 'BV1WK46eLEQ7', 'BV1WVfcYFEiX', 'BV1WWZvY4E3y',
    'BV1WjZhY1Ezo', 'BV1X5ZtYJExQ', 'BV1X5o9YHETe', 'BV1X8fMYhENU', 'BV1X8oXYpEjY', 'BV1XMZxY1Esx',
    'BV1XRPazzEcn', 'BV1XVZzYSEH4', 'BV1XXZAY7ELS', 'BV1XaZBYmEKv', 'BV1XhZBYEEp3', 'BV1XhZPYnERS',
    'BV1Y6ZJY8EeE', 'BV1Y9fwYDEyJ', 'BV1YGZqYkEc6', 'BV1YL3ez8Eo9', 'BV1YLZzYEE1Z', 'BV1YNZvYzELw',
    'BV1YQZxYjEBX', 'BV1YTZvYMECW', 'BV1YW4y1E7EV', 'BV1YcZYYeE7Y', 'BV1YqZJYaE7L', 'BV1Yv4y167q9',
    'BV1Z4ZJYuESA', 'BV1ZAodYqEy2', 'BV1ZCeb6NEfN', 'BV1ZKodYvE6V', 'BV1ZLotYREhz', 'BV1ZPqkBgEjt',
    'BV1ZYZyY3EWS', 'BV1ZeZGYnECV', 'BV1ZfoXYaERH', 'BV1ZhZHY5Eq9', 'BV1ZueH6LEa6', 'BV1ZuzfBZEw9',
    'BV1aBZbYXEft', 'BV1aCZHYKEpY', 'BV1aCZJYtENC', 'BV1aEZzYjEKU', 'BV1aHo2YeE4S', 'BV1aLLBzDEqp',
    'BV1aXj46zEgu', 'BV1anZnYAERq', 'BV1arZJYfE9h', 'BV1arZJYfELj', 'BV1auZ5YjEmj', 'BV1b2ZJYxEQc',
    'BV1b7411L71D', 'BV1b9ZHYbEwV', 'BV1bFZbYUEmW', 'BV1bFuuz2EEy', 'BV1bJZzY1Eqb', 'BV1bMo4YrEBC',
    'BV1bPZhY9EUE', 'BV1bRZWYiEmb', 'BV1bdZxYCEXi', 'BV1bj411U7mV', 'BV1byZJYoE2L', 'BV1c3ZwY4EqE',
    'BV1c4ZyYjEWs', 'BV1c6ZtYYEMy', 'BV1cGZwYdE5e', 'BV1cHHg6sEfJ', 'BV1cL3M6sEYt', 'BV1cMZwYbEU8',
    'BV1cNXYYPEyA', 'BV1cUoDYaEdb', 'BV1cXZfYREj5', 'BV1cZZ4YwEUn', 'BV1ckQmY3Erc', 'BV1cqZzYNE6X',
    'BV1cuZwYZEZr', 'BV1cv4y1y7tC', 'BV1cwZ8YTESd', 'BV1cyZnY2Ehg', 'BV1czZEYGEmg', 'BV1dEfKYcENK',
    'BV1dJZCYwEKg', 'BV1dL411w7FZ', 'BV1dLx46TExV', 'BV1dSoyYiEom', 'BV1dfoDY9E8Y', 'BV1dmZtYoEWd',
    'BV1dsXNY1EDU', 'BV1e6ZHYMEtS', 'BV1e6rKByEWA', 'BV1e9Z8YhEz3', 'BV1eAZAYDEYC', 'BV1eBZhYhEa9',
    'BV1eCZHYKE5u', 'BV1eUZvYgEVv', 'BV1eXZtYWEBE', 'BV1eqZJYaESc', 'BV1ezZJYcEaT', 'BV1fF411s7LU',
    'BV1fPzGYKEKB', 'BV1fToxYpEdg', 'BV1fgZFYJETZ', 'BV1fhZyYkEci', 'BV1fnZbYAEr1', 'BV1fpZ6YaE47',
    'BV1fyZtYfE5a', 'BV1g63YzsECq', 'BV1g9ZqYyEXq', 'BV1gCZbYJEvf', 'BV1gEodYtEBL', 'BV1gHZEY9EBY',
    'BV1gKReBSEYC', 'BV1gRZcYMEKK', 'BV1giZ4YjEAA', 'BV1gtZnYtE7B', 'BV1gu411z7nw', 'BV1gyZ6YEEuM',
    'BV1h6ZJY8ErK', 'BV1h6ZbYHETM', 'BV1hCooYFEZp', 'BV1hFZbYUEbJ', 'BV1hLZzYEEy8', 'BV1hNfTYfEhs',
    'BV1hR9RBkEeW', 'BV1hWZCYbEiY', 'BV1hWZnYcEkH', 'BV1hdZtYHEd7', 'BV1hdfFYrEPe', 'BV1hj7DzxE1B',
    'BV1hx7D6uEnt', 'BV1i142167Uy', 'BV1i3ZnYuEU6', 'BV1i7ZCYYELn', 'BV1i8ZJYTEXP', 'BV1iAZNYbE9m',
    'BV1iEZ3YkELf', 'BV1iFfTYAEGT', 'BV1iFnb6tE3U', 'BV1iMZEYYE4C', 'BV1iT4m1S7YR', 'BV1ihZhYKERR',
    'BV1ip4y1t7qq', 'BV1j4Z4YpELF', 'BV1jGZnYBEj5', 'BV1jJ4m1u7H8', 'BV1jModYNEPS', 'BV1jN4neLEZP',
    'BV1jUst66EiT', 'BV1jW4y1q71H', 'BV1jaZhYvEFz', 'BV1jdZnYuEEg', 'BV1jeho6NE6Q', 'BV1jgZEYwEH3',
    'BV1jsZ4Y9Exi', 'BV1k8411k7sW', 'BV1kBZ8YrEgL', 'BV1kCoxYZEAc', 'BV1kEZJYyE1x', 'BV1kEZJYyECh',
    'BV1kPZCYGE45', 'BV1kXZBYPEkf', 'BV1kdZBYwEUT', 'BV1keZtYiEJ1', 'BV1keZtYiEJU', 'BV1khZAYaE2V',
    'BV1kifwYAEBo', 'BV1koZ6YuE4j', 'BV1kpZHYSEpv', 'BV1kpZhYrEA7', 'BV1ksZHYEEUD', 'BV1m6ZiYeERy',
    'BV1mKo1YREkg', 'BV1mSZkYQEGf', 'BV1mXZJYHE4u', 'BV1mZsc6UE7f', 'BV1mafKYHESz', 'BV1mbZnYoEks',
    'BV1n6ZEYrEmh', 'BV1n8QzY7ExB', 'BV1nAZnYyErE', 'BV1nDy5z6Emm', 'BV1nKZnYkEnp', 'BV1nTZpYSEF4',
    'BV1nUZCYGErd', 'BV1nYZcYpEco', 'BV1ncoyYgEpG', 'BV1nposYWE5z', 'BV1nuaa6uEx8', 'BV1nwZEYqE2B',
    'BV1nyZnY2E7L', 'BV1o4Z6YfEYU', 'BV1oGYEepE53', 'BV1oLZaYNEPD', 'BV1oV411V7zA', 'BV1oko9YAE9C',
    'BV1on4q6qEdx', 'BV1onZwYJEJG', 'BV1oqZHYgExS', 'BV1ot4y1T77B', 'BV1pEZhY5EsD', 'BV1pJfNYrExn',
    'BV1pZZ7YgEZ4', 'BV1pZZ7YgEeV', 'BV1pdZ4YYETs', 'BV1ph4y1D7Lg', 'BV1pmZVYhE7M', 'BV1pqZzYPECM',
    'BV1pqZzYPEpw', 'BV1ptZAYeE31', 'BV1q2ZtYtExy', 'BV1q3ZnYuE2C', 'BV1qAZzY4Eyj', 'BV1qKZzYsECv',
    'BV1qLZ3YyEjk', 'BV1qLr6YJEh7', 'BV1qP4y1S7ML', 'BV1qS4y117am', 'BV1qWpBenE8F', 'BV1qYZJYNEK4',
    'BV1qjZpYCEGk', 'BV1qoZaYTEgi', 'BV1r6ovYQEBF', 'BV1r8DGYBEbo', 'BV1rAZrYNEA3', 'BV1rHZJYkEnK',
    'BV1rNZnYVEsm', 'BV1rNZnYVEwL', 'BV1rUZeYnEDg', 'BV1rmfKY3Eoo', 'BV1rnZJYVEwZ', 'BV1rpZhYrE6V',
    'BV1rqZtYeErC', 'BV1s54y1w7Vg', 'BV1sE9DBDELp', 'BV1sGZbY6EoW', 'BV1sLZhYjEST', 'BV1sQZ7Y5EZa',
    'BV1sSZ4YqEHx', 'BV1sSoXYzEP9', 'BV1sXZbYVETR', 'BV1sYZrYyEaM', 'BV1scZpY9EHr', 'BV1sjZhY1EMa',
    'BV1soZJYUE4x', 'BV1soZJYUEKd', 'BV1soZJYUERx', 'BV1stZAYvELe', 'BV1szePeaEhP', 'BV1t1Z4YgEcM',
    'BV1t1Z7YEEnc', 'BV1t5Hv6KENo', 'BV1t5ZvYYEJC', 'BV1t6oXY1E3n', 'BV1tDZJYXEJH', 'BV1tEZqYBE3o',
    'BV1tNfAYVEDT', 'BV1tVZcYKEyg', 'BV1tZZWYeERt', 'BV1ta411J7bP', 'BV1taVEznEfD', 'BV1tmeGeFErE',
    'BV1tnZqYTEx6', 'BV1ttZzYgEF9', 'BV1twZ4YzEmv', 'BV1twoRYJE4f', 'BV1tx4y1v7BV', 'BV1txZzYrEh2',
    'BV1txkEBiE5m', 'BV1tysi6LET8', 'BV1uBZsYTE8L', 'BV1uCZBY3E5e', 'BV1uVgczfEQr', 'BV1ubZnYoEQY',
    'BV1ubZnYoEpG', 'BV1uhZnYxEKR', 'BV1uiZsYBEZC', 'BV1um4y1U7yA', 'BV1uqoXYmEEB', 'BV1usZEYQEB8',
    'BV1uxZnYfE7u', 'BV1uyZbYUEx2', 'BV1v64y1A7jY', 'BV1v841187qC', 'BV1vDZtYRE6d', 'BV1vSZtYxEiv',
    'BV1vVZEYFEDq', 'BV1vgZrY8Eoo', 'BV1vgZzYQEbs', 'BV1vmpxejEVM', 'BV1voQfY8Ei8', 'BV1voZaYTE39',
    'BV1vqWe6EEKQ', 'BV1vvP5z8ECD', 'BV1w1421Q7V1', 'BV1w34A62Eq2', 'BV1w3ZPYNEu7', 'BV1w6ZEYrEsb',
    'BV1w6ZnYJE9t', 'BV1wFZ8YBEt4', 'BV1wGZBYaExa', 'BV1wR93Y5EUh', 'BV1wZZ4YwEgb', 'BV1who6YPEtE',
    'BV1wqoyY1EeT', 'BV1wrZ8Y6EFT', 'BV1wyZnY2E1R', 'BV1wyZnY2Epy', 'BV1x4ZqYAEYx', 'BV1x8X6YFEF8',
    'BV1x9ZJYDEot', 'BV1xJZvYFEug', 'BV1xMZbYVE5v', 'BV1xMZbYVEYg', 'BV1xYZEYjEve', 'BV1xaW96WEAG',
    'BV1xdZJYJE5N', 'BV1xgZhYSESr', 'BV1xnZcYhEa4', 'BV1xoZJYUEFD', 'BV1xsZqYeEE1', 'BV1xuZbYkEip',
    'BV1y5XrYFE1L', 'BV1y7ZJYLEMf', 'BV1yCZwYeE8H', 'BV1yD421T7Z6', 'BV1yEaJ6CEUk', 'BV1yJZHYQEAh',
    'BV1yMZ4YvEjR', 'BV1yf421q7KQ', 'BV1yocPe5ENZ', 'BV1z3ZCYqEMU', 'BV1z7Z4YbE4u', 'BV1z8Y9eZE83',
    'BV1zKZrYAEi8', 'BV1zS42197WY', 'BV1zbChYiE38', 'BV1zhZHY5Edb', 'BV1zmZ8YvEFC', 'BV1zsZJY6EJm',
    'BV1zvoQYFEbi', 'BV1zyZ8YqEFR',
  ];

  const MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
  ];

  const CONTACT_RE = /[@＠]|https?:\/\/|\bwww\.|\b1[3-9]\d{9}\b|手机号|身份证|家庭住址|加我微信|加QQ群/i;
  const HARD_GATE_CODES = new Set([-352, -412, -101, -403]);
  const SEARCH_ORDERS = ['click', 'dm', 'stow', 'scores'];

  const ERA_PRESETS = {
    pre_llm: {
      label: 'pre_llm (<=2022)',
      minPubdate: 1514736000,   // 2018-01-01 00:00:00 CST
      maxPubdate: 1672502399,   // 2022-12-31 23:59:59 CST
      useRegionFallback: false,
      windows: [
        { name: 'y2022', begin: 1640966400, end: 1672502399 },
        { name: 'y2021', begin: 1609430400, end: 1640966399 },
        { name: 'y2020', begin: 1577808000, end: 1609430399 },
        { name: 'y2019', begin: 1546272000, end: 1577807999 },
      ],
      yearHints: ['2022', '2021', '2020', '2019'],
      orders: ['click', 'dm', 'stow', 'scores'],
      boostNews: false,
    },
    early_llm: {
      label: 'early_llm (2023-2024)',
      minPubdate: 1672502400,   // 2023-01-01 00:00:00 CST
      maxPubdate: 1735660799,   // 2024-12-31 23:59:59 CST
      useRegionFallback: false,
      windows: [
        { name: 'y2024', begin: 1704038400, end: 1735660799 },
        { name: 'y2023', begin: 1672502400, end: 1704038399 },
      ],
      yearHints: ['2024', '2023'],
      orders: ['click', 'dm', 'stow', 'scores'],
      boostNews: false,
    },
    current_2026_news: {
      label: 'current_2026 + news boost',
      minPubdate: 1767196800,   // 2026-01-01 00:00:00 CST
      maxPubdate: Math.floor(Date.now() / 1000) + 86400,
      useRegionFallback: true,
      windows: [
        { name: 'y2026', begin: 1767196800, end: Math.floor(Date.now() / 1000) + 86400 },
      ],
      yearHints: ['', '2026'],
      orders: ['pubdate', 'click', 'dm'],
      boostNews: true,
    },
  };

  const NEWS_KEYWORDS = [
    '行业观察', '财报解读', '科技动态', '民生热点', '供应链分析', '新闻简报',
    '经济数据', '政策解读', '金融市场观察', '国际局势分析', '芯片半导体', 'AI行业新闻',
    '新能源行业', '互联网大厂', '消费趋势分析', '城市规划建设', '商业评论', '宏观经济',
  ];

  const PARTITIONS = [
    { name: '动画', rid: 1, kw: ['动画解析', '手书', '独立动画', '定格动画', '音MAD', '原画作画', '作画细节', '动画短片'] },
    { name: '番剧国创', rid: 167, kw: ['新番吐槽', '国创动画', '经典老番', '番剧杂谈', '动漫盘点', '剧场版解析', '补番推荐', '番剧结局'] },
    { name: '游戏', rid: 4, kw: ['单机攻略', '独立游戏测评', '速通', '怀旧老游戏', '剧情考据', '游戏实况', 'BOSS打法', 'MOD分享'] },
    { name: '知识', rid: 36, kw: ['科普公开课', '历史长视频', '物理讲解', '经济学分析', '冷知识', '法律科普', '纪录考据', '数学直觉'] },
    { name: '科技数码', rid: 188, kw: ['显卡实测', '开源AI部署', 'NAS教程', '机械键盘', '手机续航', '路由器评测', 'Linux折腾', '装机指南'] },
    { name: '生活', rid: 160, kw: ['独居Vlog', '手工DIY', '日常整活', '出租屋改造', '极简收纳', '农村生活', '周末vlog', '真实日常'] },
    { name: '美食', rid: 211, kw: ['家常菜教程', '美食探店', '烘焙实操', '街头小吃', '酱料复刻', '减脂餐测评', '做饭记录', '地方特产'] },
    { name: '影视', rid: 181, kw: ['电影拉片', '悬疑剧拆解', '老剧重温', '纪录片解说', '武侠打戏', '文艺片解读', '国产剧混剪', '演技赏析'] },
    { name: '音乐', rid: 3, kw: ['编曲教学', '现场Live', '吉他指弹', '乐评盘点', '古典乐分析', '翻唱混音', '演奏现场', '老歌翻唱'] },
    { name: '鬼畜', rid: 119, kw: ['鬼畜调教', '人力Vocaloid', '鬼畜剧场', '全明星二创', '方言鬼畜', '文艺复兴', '鬼畜经典', '高能剪辑'] },
    { name: '娱乐', rid: 5, kw: ['综艺名场面', '采访访谈', '脱口秀', '晚会现场', '幕后花絮', '明星互动', '访谈实录', '舞台直拍'] },
    { name: '资讯', rid: 202, kw: NEWS_KEYWORDS },
    { name: '汽车', rid: 223, kw: ['新能源续航实测', '二手车整备', '底盘对比', '自驾游路书', '智驾实测', '老车翻新', '试驾感受', '保养避坑'] },
    { name: '运动', rid: 234, kw: ['骑行记录', '羽毛球教学', '马拉松装备', '徒步露营', '力量训练', '滑雪教程', '乒乓球实战', '跑步打卡'] },
    { name: '动物', rid: 217, kw: ['流浪猫救助', '野保观察', '水族造景', '养狗训练', '萌宠日常', '鸟类摄影', '猫咪日记', '爬虫饲养'] },
    { name: '时尚舞蹈', rid: 155, kw: ['街舞编舞', '国风穿搭', '宅舞翻跳', '球鞋测评', '男生穿搭', '民族舞', '穿搭思路', '复古穿搭'] },
  ];

  const MD5_S = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
  ];
  const MD5_K = Array.from({ length: 64 }, (_, i) => Math.floor(Math.abs(Math.sin(i + 1)) * 4294967296) >>> 0);

  function md5(str) {
    const raw = new TextEncoder().encode(String(str));
    const bitLen = raw.length * 8;
    const paddedLen = (((raw.length + 8) >>> 6) + 1) << 6;
    const buf = new Uint8Array(paddedLen);
    buf.set(raw);
    buf[raw.length] = 0x80;
    const dv = new DataView(buf.buffer);
    dv.setUint32(paddedLen - 8, bitLen >>> 0, true);
    dv.setUint32(paddedLen - 4, Math.floor(bitLen / 4294967296) >>> 0, true);

    let a0 = 0x67452301;
    let b0 = 0xefcdab89;
    let c0 = 0x98badcfe;
    let d0 = 0x10325476;

    for (let off = 0; off < paddedLen; off += 64) {
      let A = a0, B = b0, C = c0, D = d0;
      for (let i = 0; i < 64; i++) {
        let F, g;
        if (i < 16) {
          F = (B & C) | ((~B) & D);
          g = i;
        } else if (i < 32) {
          F = (D & B) | ((~D) & C);
          g = (5 * i + 1) & 15;
        } else if (i < 48) {
          F = B ^ C ^ D;
          g = (3 * i + 5) & 15;
        } else {
          F = C ^ (B | (~D));
          g = (7 * i) & 15;
        }
        const word = dv.getUint32(off + g * 4, true);
        const sum = (A + F + MD5_K[i] + word) >>> 0;
        const rot = ((sum << MD5_S[i]) | (sum >>> (32 - MD5_S[i]))) >>> 0;
        A = D;
        D = C;
        C = B;
        B = (B + rot) >>> 0;
      }
      a0 = (a0 + A) >>> 0;
      b0 = (b0 + B) >>> 0;
      c0 = (c0 + C) >>> 0;
      d0 = (d0 + D) >>> 0;
    }

    const out = [a0, b0, c0, d0];
    let hex = '';
    for (const w of out) {
      for (let b = 0; b < 4; b++) {
        hex += ((w >>> (b * 8)) & 0xff).toString(16).padStart(2, '0');
      }
    }
    return hex;
  }

  function encWbi(params, imgKey, subKey) {
    const orig = imgKey + subKey;
    const mixinKey = MIXIN_KEY_ENC_TAB.map((n) => orig[n] || '').join('').slice(0, 32);
    const full = { ...params, wts: Math.floor(Date.now() / 1000) };
    const query = Object.keys(full)
      .sort()
      .map((k) => {
        const val = String(full[k]).replace(/[!'()*]/g, '');
        return `${encodeURIComponent(k)}=${encodeURIComponent(val)}`;
      })
      .join('&');
    const w_rid = md5(query + mixinKey);
    return `${query}&w_rid=${w_rid}`;
  }

  function eraFromTs(ts) {
    const n = Number(ts) || 0;
    if (n <= 0) return 'current_2026';
    const yr = new Date((n + 8 * 3600) * 1000).getUTCFullYear();
    if (yr <= 2022) return 'pre_llm';
    if (yr <= 2024) return 'early_llm';
    if (yr === 2025) return 'mid_llm';
    return 'current_2026';
  }

  function extractKeyFromUrl(url) {
    if (!url) return '';
    const m = String(url).match(/\/([^\/?#]+)\.\w+(?:[?#]|$)/);
    return m ? m[1] : '';
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function cleanText(raw) {
    if (typeof raw !== 'string') return '';
    const txt = raw.replace(/\s+/g, ' ').trim();
    if (!txt || CONTACT_RE.test(txt)) return '';
    return txt;
  }

  function triggerDownload(filename, lines) {
    const blob = new Blob([lines.join('\n') + '\n'], { type: 'application/x-ndjson;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2500);
  }

  function readStorageSet() {
    const out = new Set();
    try {
      if (typeof localStorage !== 'undefined') {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
          const arr = JSON.parse(raw);
          if (Array.isArray(arr)) {
            for (const b of arr) {
              if (typeof b === 'string' && b.trim()) out.add(b.trim());
            }
          }
        }
      }
    } catch { /* ignore */ }
    return out;
  }

  function unionWriteStorageSet(localSet) {
    try {
      if (typeof localStorage !== 'undefined') {
        const disk = readStorageSet();
        for (const b of localSet) disk.add(b);
        for (const b of disk) localSet.add(b);
        localStorage.setItem(STORAGE_KEY, JSON.stringify([...disk]));
      }
    } catch { /* ignore */ }
  }

  function buildSchedule(boostNews) {
    const sched = [];
    const newsPart = PARTITIONS.find((p) => p.name === '资讯');
    for (const p of PARTITIONS) {
      sched.push(p);
      if (boostNews && p.name !== '资讯' && newsPart && sched.length % 5 === 0) {
        sched.push(newsPart);
      }
    }
    if (boostNews && newsPart) {
      sched.push(newsPart, newsPart);
    }
    return sched;
  }

  async function runBiliV04Wave2(options = {}) {
    const presetName = ERA_PRESETS[options.preset] ? options.preset : 'pre_llm';
    const presetCfg = ERA_PRESETS[presetName];
    const wave = String(options.wave || ('w2_' + presetName)).replace(/[^a-zA-Z0-9_-]/g, '');
    const runTag = String(options.runTag || Math.random().toString(36).slice(2, 6)).replace(/[^a-zA-Z0-9_-]/g, '') || 'tab0';
    const targetRows = Number(options.targetRows ?? 180000);
    const maxMinutes = Number(options.maxMinutes ?? 180);
    const maxRounds = Number(options.maxRounds ?? 60);
    const flushEvery = Number(options.flushEvery ?? 5000);
    const maxPerVideo = Number(options.maxPerVideo ?? 800);
    const maxPagesPerVideo = Number(options.maxPagesPerVideo ?? 35);
    const concurrency = Math.max(1, Math.min(6, Number(options.concurrency ?? 3)));
    const delayMinMs = Math.max(100, Number(options.delayMinMs ?? 240));
    const delayMaxMs = Math.max(delayMinMs, Number(options.delayMaxMs ?? 420));
    const captureDate = new Date().toISOString().slice(0, 10);
    const deadline = Date.now() + maxMinutes * 60_000;

    const seenVideos = new Set(INITIAL_SKIP_BVIDS);
    if (Array.isArray(options.skipBvids)) {
      for (const b of options.skipBvids) {
        if (typeof b === 'string' && b.trim()) seenVideos.add(b.trim());
      }
    }
    unionWriteStorageSet(seenVideos);

    const discoveredQueueSet = new Set(seenVideos);
    const seenComments = new Set();
    const buffer = [];
    const stopSignal = { stopped: false, reason: null, code: null };

    let bc = null;
    if (typeof BroadcastChannel !== 'undefined') {
      try {
        bc = new BroadcastChannel(CHANNEL_NAME);
        bc.onmessage = (ev) => {
          const bvid = ev?.data?.bvid;
          if (typeof bvid === 'string' && bvid) {
            seenVideos.add(bvid);
            discoveredQueueSet.add(bvid);
          }
        };
      } catch { bc = null; }
    }
    const onStorage = (ev) => {
      if (ev && ev.key === STORAGE_KEY && ev.newValue) {
        try {
          const arr = JSON.parse(ev.newValue);
          if (Array.isArray(arr)) {
            for (const b of arr) {
              if (typeof b === 'string') {
                seenVideos.add(b);
                discoveredQueueSet.add(b);
              }
            }
          }
        } catch {}
      }
    };
    if (typeof root?.addEventListener === 'function') {
      root.addEventListener('storage', onStorage);
    }

    const state = {
      wave,
      preset: presetName,
      eraLabel: presetCfg.label,
      runTag,
      startedAt: new Date().toISOString(),
      running: true,
      done: false,
      status: 'running',
      concurrency,
      maxRounds,
      maxMinutes,
      targetRows,
      preSkippedBvids: seenVideos.size,
      totalRows: 0,
      chunksDownloaded: 0,
      downloadedFiles: [],
      videosVisited: 0,
      crossTabSkipped: 0,
      rejectedByEra: 0,
      byPartition: {},
      byEra: {},
      lastBvid: null,
      stopCode: null,
      searchSkipped: 0,
      error: null,
    };
    if (root && typeof root === 'object') {
      root.__biliV04State = state;
      root.__biliV04Controller = {
        stop() {
          stopSignal.stopped = true;
          stopSignal.reason = 'manual_stop';
        },
      };
    }

    const claimBvid = (bvid) => {
      if (!bvid || seenVideos.has(bvid)) {
        state.crossTabSkipped += 1;
        return false;
      }
      const disk = readStorageSet();
      for (const b of disk) seenVideos.add(b);
      if (seenVideos.has(bvid)) {
        state.crossTabSkipped += 1;
        return false;
      }
      seenVideos.add(bvid);
      discoveredQueueSet.add(bvid);
      try {
        if (typeof localStorage !== 'undefined') {
          disk.add(bvid);
          localStorage.setItem(STORAGE_KEY, JSON.stringify([...disk]));
        }
      } catch {}
      if (bc) {
        try { bc.postMessage({ type: 'claim', bvid }); } catch {}
      }
      return true;
    };

    const flush = (force = false) => {
      if (!buffer.length || (!force && buffer.length < flushEvery)) return;
      state.chunksDownloaded += 1;
      const fname = `bili_v04_${wave}_${runTag}_chunk_${String(state.chunksDownloaded).padStart(3, '0')}.jsonl`;
      const chunk = buffer.splice(0, buffer.length);
      triggerDownload(fname, chunk);
      state.downloadedFiles.push({ file: fname, rows: chunk.length });
      unionWriteStorageSet(seenVideos);
    };

    const shouldStop = () => stopSignal.stopped || Date.now() >= deadline || state.totalRows >= targetRows;

    const apiGet = async (url) => {
      if (shouldStop()) return { gate: false, cancelled: true, code: 0, json: null };
      const jitter = delayMinMs + Math.floor(Math.random() * (delayMaxMs - delayMinMs + 1));
      await sleep(jitter);
      if (shouldStop()) return { gate: false, cancelled: true, code: 0, json: null };
      const resp = await fetch(url, { credentials: 'include' });
      if (resp.status === 412) {
        stopSignal.stopped = true;
        stopSignal.reason = 'stopped_on_gate';
        stopSignal.code = 412;
        return { gate: true, code: 412, json: null };
      }
      if (!resp.ok) {
        return { gate: false, code: resp.status, json: null };
      }
      const j = await resp.json();
      if (j && HARD_GATE_CODES.has(j.code)) {
        stopSignal.stopped = true;
        stopSignal.reason = 'stopped_on_gate';
        stopSignal.code = j.code;
        return { gate: true, code: j.code, json: j };
      }
      return { gate: false, code: j?.code ?? 0, json: j };
    };

    const isInPresetEra = (pubdate) => {
      const p = Number(pubdate || 0);
      if (p <= 0) return false;
      return p >= presetCfg.minPubdate && p <= presetCfg.maxPubdate;
    };

    try {
      const nav = await apiGet('https://api.bilibili.com/x/web-interface/nav');
      if (nav.gate) {
        state.status = 'stopped_on_gate';
        state.stopCode = nav.code;
        return finish();
      }
      const navData = nav.json?.data;
      if (!navData?.isLogin) {
        state.status = 'login_required';
        return finish();
      }
      const imgKey = extractKeyFromUrl(navData.wbi_img?.img_url);
      const subKey = extractKeyFromUrl(navData.wbi_img?.sub_url);

      const addRow = (video, rawReply, level, parentId, sortLabel) => {
        if (!rawReply || !rawReply.rpid) return false;
        const rpid = String(rawReply.rpid);
        if (seenComments.has(rpid)) return false;
        const txt = cleanText(rawReply.content?.message);
        if (!txt) return false;
        seenComments.add(rpid);
        const ts = Number(rawReply.ctime || 0) || null;
        const era = eraFromTs(ts || video.pubdate || 0);
        const rec = {
          bvid: video.bvid,
          partition: video.partition,
          video_title: video.title,
          pubdate: video.pubdate,
          comment_level: level,
          parent_id: parentId || null,
          text: txt,
          posted_ts: ts,
          like_count: typeof rawReply.like === 'number' ? rawReply.like : null,
          is_pinned: Boolean(rawReply.card_label?.some?.((c) => /置顶/.test(c?.text_content || ''))),
          sort: sortLabel,
          capture_date: captureDate,
        };
        buffer.push(JSON.stringify(rec));
        state.totalRows += 1;
        state.byPartition[video.partition] = (state.byPartition[video.partition] || 0) + 1;
        state.byEra[era] = (state.byEra[era] || 0) + 1;
        if (buffer.length >= flushEvery) flush(true);
        return true;
      };

      const fetchVideoComments = async (video) => {
        if (shouldStop()) return 'stop';
        if (!claimBvid(video.bvid)) return 'skip';
        let aid = video.aid;
        let pubdate = Number(video.pubdate || 0) || null;
        let title = video.title || '';
        if (!aid || !pubdate) {
          const vw = await apiGet(`https://api.bilibili.com/x/web-interface/view?bvid=${encodeURIComponent(video.bvid)}`);
          if (vw.gate) {
            state.stopCode = vw.code;
            return 'gate';
          }
          if (vw.cancelled) return 'stop';
          aid = aid || vw.json?.data?.aid;
          pubdate = pubdate || Number(vw.json?.data?.pubdate || 0) || null;
          title = title || (vw.json?.data?.title || '');
        }
        if (!aid || !isInPresetEra(pubdate)) {
          state.rejectedByEra += 1;
          return 'skip';
        }
        const vmeta = {
          bvid: video.bvid,
          aid,
          partition: video.partition,
          title: String(title).replace(/\s+/g, ' ').trim().slice(0, 260),
          pubdate,
        };
        state.videosVisited += 1;
        state.lastBvid = vmeta.bvid;

        let videoCount = 0;
        let offset = '';
        let useClassic = false;
        let pn = 1;
        let pageGuard = 0;

        while (pageGuard < maxPagesPerVideo && videoCount < maxPerVideo && !shouldStop()) {
          pageGuard += 1;
          let res;
          if (!useClassic && imgKey && subKey) {
            const qs = encWbi({
              oid: aid,
              type: 1,
              mode: 3,
              plat: 1,
              web_location: '1315875',
              pagination_str: JSON.stringify({ offset }),
            }, imgKey, subKey);
            res = await apiGet(`https://api.bilibili.com/x/v2/reply/wbi/main?${qs}`);
            if (res.gate) {
              state.stopCode = res.code;
              return 'gate';
            }
            if (res.cancelled) return 'stop';
            if (res.code !== 0 || !res.json?.data) {
              useClassic = true;
              continue;
            }
          } else {
            res = await apiGet(`https://api.bilibili.com/x/v2/reply?type=1&oid=${aid}&pn=${pn}&ps=20&sort=2`);
            if (res.gate) {
              state.stopCode = res.code;
              return 'gate';
            }
            if (res.cancelled) return 'stop';
            if (res.code !== 0 || !res.json?.data) break;
            pn += 1;
          }

          const d = res.json.data;
          const topList = [];
          if (pageGuard === 1) {
            if (Array.isArray(d.top_replies)) topList.push(...d.top_replies);
            if (d.top?.upper) topList.push(d.top.upper);
          }
          const mainList = Array.isArray(d.replies) ? d.replies : [];
          const combined = [...topList, ...mainList];
          if (!combined.length) break;

          const needReplyThreads = [];
          for (const item of combined) {
            if (videoCount >= maxPerVideo || shouldStop()) break;
            if (addRow(vmeta, item, 'top', null, useClassic ? 'classic_hot' : 'wbi_main')) {
              videoCount += 1;
            }
            if (Array.isArray(item.replies)) {
              for (const sub of item.replies) {
                if (videoCount >= maxPerVideo || shouldStop()) break;
                if (addRow(vmeta, sub, 'reply', String(item.rpid), useClassic ? 'classic_embedded' : 'wbi_embedded')) {
                  videoCount += 1;
                }
              }
            }
            if ((item.rcount || 0) > 3 && needReplyThreads.length < 6) {
              needReplyThreads.push(item.rpid);
            }
          }

          for (const rootRpid of needReplyThreads) {
            if (videoCount >= maxPerVideo || shouldStop()) break;
            const subRes = await apiGet(`https://api.bilibili.com/x/v2/reply/reply?type=1&oid=${aid}&root=${rootRpid}&pn=1&ps=20`);
            if (subRes.gate) {
              state.stopCode = subRes.code;
              return 'gate';
            }
            if (subRes.cancelled) return 'stop';
            for (const sub of (subRes.json?.data?.replies || [])) {
              if (videoCount >= maxPerVideo || shouldStop()) break;
              if (addRow(vmeta, sub, 'reply', String(rootRpid), 'reply_p1')) {
                videoCount += 1;
              }
            }
          }

          if (!useClassic) {
            const nextOff = d.cursor?.pagination_reply?.next_offset;
            if (!nextOff || d.cursor?.is_end) break;
            offset = nextOff;
          } else {
            if (mainList.length < 20) break;
          }
        }
        unionWriteStorageSet(seenVideos);
        return 'ok';
      };

      const schedule = buildSchedule(presetCfg.boostNews);
      const queues = new Map(PARTITIONS.map((p) => [p.name, []]));

      const enqueueVideo = (q, v, partitionName) => {
        if (!v || !v.bvid || discoveredQueueSet.has(v.bvid) || seenVideos.has(v.bvid)) return;
        const pub = Number(v.pubdate || 0);
        if (pub > 0 && !isInPresetEra(pub)) {
          state.rejectedByEra += 1;
          return;
        }
        if (typeof v.review === 'number' && v.review < 10) return;
        discoveredQueueSet.add(v.bvid);
        q.push({
          bvid: v.bvid,
          aid: v.aid || v.id,
          title: String(v.title || '').replace(/<[^>]+>/g, ''),
          pubdate: pub || null,
          partition: partitionName,
        });
      };

      const runWorker = async (workerId) => {
        for (let round = 0; round < maxRounds && !shouldStop(); round++) {
          for (let sIdx = workerId; sIdx < schedule.length && !shouldStop(); sIdx += concurrency) {
            const part = schedule[sIdx];
            const q = queues.get(part.name);

            if (q.length < 4) {
              if (presetCfg.useRegionFallback) {
                const regionPn = (round % 15) + 1;
                const dyn = await apiGet(`https://api.bilibili.com/x/web-interface/dynamic/region?rid=${part.rid}&ps=25&pn=${regionPn}`);
                if (dyn.gate) { state.stopCode = dyn.code; return; }
                for (const v of (dyn.json?.data?.archives || [])) enqueueVideo(q, v, part.name);
              }

              if (imgKey && subKey && !shouldStop()) {
                const win = presetCfg.windows[(round + sIdx) % presetCfg.windows.length];
                const rawKw = part.kw[(round + sIdx) % part.kw.length];
                const hint = presetCfg.yearHints[(round + sIdx) % presetCfg.yearHints.length];
                const keyword = hint ? `${hint} ${rawKw}` : rawKw;
                const order = presetCfg.orders[Math.floor((round + sIdx) / part.kw.length) % presetCfg.orders.length];
                const page = (Math.floor(round / Math.max(1, presetCfg.orders.length)) % 10) + 1;
                const qs = encWbi({
                  search_type: 'video',
                  keyword,
                  order,
                  page,
                  pubtime_begin_s: win.begin,
                  pubtime_end_s: win.end,
                }, imgKey, subKey);
                const sr = await apiGet(`https://api.bilibili.com/x/web-interface/wbi/search/type?${qs}`);
                if (sr.gate) { state.stopCode = sr.code; return; }
                if (sr.code === 0 && Array.isArray(sr.json?.data?.result)) {
                  const resList = [...sr.json.data.result].sort((a, b) => (Number(b.review || 0) - Number(a.review || 0)));
                  for (const v of resList) enqueueVideo(q, v, part.name);
                } else if (!sr.cancelled) {
                  state.searchSkipped += 1;
                }
              }
            }

            for (let take = 0; take < 3 && q.length && !shouldStop(); take++) {
              const vid = q.shift();
              const outcome = await fetchVideoComments(vid);
              if (outcome === 'gate' || outcome === 'stop') return;
            }
          }
        }
      };

      await Promise.all(Array.from({ length: concurrency }, (_, id) => runWorker(id)));

      if (stopSignal.reason) {
        return finish(stopSignal.reason);
      }
      return finish(state.totalRows >= targetRows ? 'target_reached' : Date.now() >= deadline ? 'time_budget_reached' : 'exhausted');
    } catch (err) {
      state.error = String(err?.message || err).slice(0, 220);
      return finish('error');
    }

    function finish(finalStatus) {
      if (finalStatus) state.status = finalStatus;
      if (stopSignal.code && !state.stopCode) state.stopCode = stopSignal.code;
      flush(true);
      unionWriteStorageSet(seenVideos);
      if (bc) { try { bc.close(); } catch {} }
      if (typeof root?.removeEventListener === 'function') {
        root.removeEventListener('storage', onStorage);
      }
      state.running = false;
      state.done = true;
      state.finishedAt = new Date().toISOString();
      return state;
    }
  }

  if (root && typeof root === 'object') {
    root.runBiliV04Wave2 = runBiliV04Wave2;
    root.__biliV04Wave2Start = function (opts) {
      if (root.__biliV04State && root.__biliV04State.running) {
        return { started: false, reason: 'already_running', state: root.__biliV04State };
      }
      root.__biliV04Promise = runBiliV04Wave2(opts || {});
      return {
        started: true,
        preset: (opts && opts.preset) || 'pre_llm',
        concurrency: Number((opts || {}).concurrency ?? 3),
      };
    };
    root.__biliV04Start = root.__biliV04Wave2Start;
    root.__biliV04Stop = function () {
      if (root.__biliV04Controller) {
        root.__biliV04Controller.stop();
      }
      return root.__biliV04State || null;
    };
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { md5, encWbi, runBiliV04Wave2, ERA_PRESETS, PARTITIONS, INITIAL_SKIP_BVIDS, buildSchedule };
  }
})(typeof window !== 'undefined' ? window : globalThis);
